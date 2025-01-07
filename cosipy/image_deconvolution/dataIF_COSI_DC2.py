import numpy as np
import cupy as cp
from numba import jit, prange
from tqdm.autonotebook import tqdm
import astropy.units as u

import logging
logger = logging.getLogger(__name__)

import warnings

from histpy import Histogram, Axes

from cosipy.response import FullDetectorResponse

from .image_deconvolution_data_interface_base import ImageDeconvolutionDataInterfaceBase

def tensordot_sparse(A, A_unit, B, axes):
    """
    perform a tensordot operation on A and B.  A is sparse
    and so does not carry a unit; rather it must be passed
    as a separate argument.  B is a normaly Quantity. Return
    a proper Quantity as the result.
    """
    dotprod = np.tensordot(A, B.value, axes=axes)
    return u.Quantity(dotprod, unit= A_unit * B.unit, copy=False)

def tensordot_resp(A, response, axes):
    """
    perform a tensordot operation between A, a Quantity array, and
    response, an image response Histogram whose contents have been
    copied to the CUDA device. We assume a reference to the device
    response array is attached to the response Histogram.
    Return a proper Quantity as the result.
    """
    dotprod = cp.tensordot(cp.array(A), response.cu_array, axes=axes).get()
    return u.Quantity(dotprod, unit= A.unit * response.unit, copy=False)

class DataIF_COSI_DC2(ImageDeconvolutionDataInterfaceBase):
    """
    A subclass of ImageDeconvolutionDataInterfaceBase for the COSI data challenge 2.
    """

    def __init__(self, name = None, dtype = None):

        ImageDeconvolutionDataInterfaceBase.__init__(self, name)

        self._image_response = None # histpy.Histogram (dense)

        # None if using Galactic CDS, mandotary if using local CDS
        self._coordsys_conv_matrix = None

        # optional
        self.is_miniDC2_format = False #should be removed in the future

    @classmethod
    def load(cls, name, event_binned_data, dict_bkg_binned_data, rsp, coordsys_conv_matrix = None, is_miniDC2_format = False, dtype = None):
        """
        Load data

        Parameters
        ----------
        name : str
            The name of data
        event_binned_data : :py:class:`histpy.Histogram`
            Event histogram
        dict_bkg_binned_data : dict
            Background models as {background_model_name: :py:class:`histpy.Histogram`}
        rsp : :py:class:`histpy.Histogram` or :py:class:`cosipy.response.FullDetectorResponse`
            Response
        coordsys_conv_matrix : :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix`, default False
            Coordsys conversion matrix
        is_miniDC2_format : bool, default False
            Whether the file format is for mini-DC2. It will be removed in the future.
        dtype : np.dtype
            Data type to be used for elements of loaded event, bkg_models,
            response, and coordsys_conv_matrix
        Returns
        -------
        :py:class:`cosipy.image_deconvolution.DataIF_COSI_DC2`
            An instance of DataIF_COSI_DC2 containing the input data set
        """

        new = cls(name)

        new._event = event_binned_data.to_dense()
        new._event =  new._event.to(unit=u.dimensionless_unscaled,
                                    update=False, copy=False)

        # we don't need overflow tracking of binned data for RL
        new._event.track_overflow(False)

        # coerce data type of event
        new._event = new._event.astype(dtype, copy = False)

        new._bkg_models = dict_bkg_binned_data

        for key in new._bkg_models:
            if new._bkg_models[key].is_sparse:
                new._bkg_models[key] = new._bkg_models[key].to_dense()

            # we don't need overflow tracking of bkg binned data for RL
            new._bkg_models[key].track_overflow(False)
            # coerce data type of bkg model
            new._bkg_models[key] = new._bkg_models[key].astype(dtype, copy = False)

            new._summed_bkg_models[key] = np.sum(new._bkg_models[key])

        # coerce data type of conversion matrix
        new._coordsys_conv_matrix = coordsys_conv_matrix
        if new._coordsys_conv_matrix is not None:
            new._coordsys_conv_matrix = new._coordsys_conv_matrix.astype(dtype, copy = False)

        # Enable sparse reshape caching to accelerate tensordot calls.
        # CoordSysConvMatrix disables overflow tracking, so
        # .contents is a view of the entire matrix.
        if new._coordsys_conv_matrix is not None and \
           new._coordsys_conv_matrix.is_sparse:
            new._coordsys_conv_matrix.contents.enable_caching()

        new.is_miniDC2_format = is_miniDC2_format

        if isinstance(rsp, FullDetectorResponse):
            logger.info('Loading the response matrix onto your computer memory...')
            new._load_full_detector_response_on_memory(rsp, is_miniDC2_format, dtype)
            logger.info('Finished')
        elif isinstance(rsp, Histogram):
            new._image_response = rsp
            new._image_response.track_overflow(False) # not needed for RL
            new._image_response = new._image_response.astype(dtype, copy = False) # coerce type of response

        # We modify the axes in event, bkg_models, response. This is only for DC2.
        new._modify_axes()

        # copy response to device and attach to this object for later reference
        new._image_response.cu_array = cp.array(new._image_response.contents)

        for key in new._bkg_models:
            if new._bkg_models[key].is_sparse:
                new._bkg_models[key] = new._bkg_models[key].to_dense()

            # we don't need overflow tracking of bkg binned data for RL
            new._bkg_models[key].track_overflow(False)
            # coerce data type of bkg model
            new._bkg_models[key] = new._bkg_models[key].astype(dtype, copy = False)

            # copy bkg model to device and attach to this object for later reference
            new._bkg_models[key].cu_array = cp.array(new._bkg_models[key].contents.ravel())

            new._summed_bkg_models[key] = np.sum(new._bkg_models[key])

        new._data_axes = new._event.axes

        if new._coordsys_conv_matrix is None:
            axes = (new._image_response.axes['NuLambda'].copy(), new._image_response.axes['Ei']) # will mutate axes[0]
            axes[0].label = 'lb'
            # The gamma-ray direction of pre-computed response in DC2 is in the galactic coordinate, not in the local coordinate.
            # Actually, it is labeled as 'NuLambda'. So I replace it with 'lb'.
            new._model_axes = Axes(axes)
        else:
            new._model_axes = Axes((new._coordsys_conv_matrix.axes['lb'], new._image_response.axes['Ei']))

        new._calc_exposure_map()

        return new

    def _modify_axes(self):
        """
        Modify the axes of data. This method will be removed in the future.
        """

        warnings.warn("Note that _modify_axes() in DataIF_COSI_DC2 was implemented for a temporary use. It will be removed in the future.", DeprecationWarning)

        if self._coordsys_conv_matrix is None:
            axis_name = ['Em', 'Phi', 'PsiChi']

        elif self._coordsys_conv_matrix.binning_method == 'Time':
            axis_name = ['Time', 'Em', 'Phi', 'PsiChi']

        elif self._coordsys_conv_matrix.binning_method == 'ScAtt':
            axis_name = ['ScAtt', 'Em', 'Phi', 'PsiChi']

        for name in axis_name:

            logger.info(f"... checking the axis {name} of the event and background files...")

            event_edges, event_unit = self._event.axes[name].edges, self._event.axes[name].unit

            for key in self._bkg_models:

                bkg_edges, bkg_unit = self._bkg_models[key].axes[name].edges, self._bkg_models[key].axes[name].unit

                if np.all(event_edges == bkg_edges):
                    logger.info(f"    --> pass (edges)")
                else:
                    logger.error(f"Warning: the edges of the axis {name} are not consistent between the event and the background model {key}!")
                    logger.error(f"         event      : {event_edges}")
                    logger.error(f"         background : {bkg_edges}")
                    raise ValueError

        # check the axes of the event/response files.
        # Note that currently (2023-08-29) no unit is stored in the binned data. So only the edges are compared. This should be modified in the future.

        axis_name = ['Em', 'Phi', 'PsiChi']

        for name in axis_name:

            logger.info(f"...checking the axis {name} of the event and response files...")

            event_edges, event_unit = self._event.axes[name].edges, self._event.axes[name].unit
            response_edges, response_unit = self._image_response.axes[name].edges, self._image_response.axes[name].unit

#            if type(response_edges) == u.quantity.Quantity and self.is_miniDC2_format == True:
            if event_unit is None and response_unit is not None and self.is_miniDC2_format == True: # this is only for the old data in the miniDC2 challenge. I will remove them in the near future (or in the final dataIF).
                response_edges = response_edges.value

            if np.all(event_edges == response_edges):
                logger.info(f"    --> pass (edges)")
            else:
                logger.error(f"Warning: the edges of the axis {name} are not consistent between the event and background!")
                logger.error(f"        event      : {event_edges}")
                logger.error(f"        response : {response_edges}")
                raise ValueError

        if self._coordsys_conv_matrix is None:
            axes_cds = Axes((self._image_response.axes["Em"], \
                             self._image_response.axes["Phi"], \
                             self._image_response.axes["PsiChi"]),
                            copy_axes=False)
        else:
            axes_cds = Axes((self._event.axes[0], \
                             self._image_response.axes["Em"], \
                             self._image_response.axes["Phi"], \
                             self._image_response.axes["PsiChi"]),
                            copy_axes=False)

        self._event = Histogram(axes_cds,
                                contents = self._event.contents,
                                unit = self._event.unit,
                                copy_contents = False) # overwite axes of existing Histogram

        for key in self._bkg_models:
            bkg_model = self._bkg_models[key]
            self._bkg_models[key] = Histogram(axes_cds,
                                              contents = bkg_model.contents,
                                              unit = bkg_model.unit,
                                              copy_contents = False) # overwite axes of existing Histogram

        logger.info(f"The axes in the event and background files are redefined. Now they are consistent with those of the response file.")

        return True

    def _load_full_detector_response_on_memory(self, full_detector_response, is_miniDC2_format, dtype):
        """
        Load a response file on the computer memory.
        """

        axes_image_response = Axes((full_detector_response.axes["NuLambda"],
                                    full_detector_response.axes["Ei"],
                                    full_detector_response.axes["Em"],
                                    full_detector_response.axes["Phi"],
                                    full_detector_response.axes["PsiChi"]),
                                   copy_axes=False)

        if is_miniDC2_format:
            npix = axes_image_response["NuLambda"].npix
            slices = [ np.sum(full_detector_response[ipix].to_dense(), axis = (4,5)) for ipix in tqdm(range(npix)) ] #Ei, Em, Phi, ChiPsi
            contents = np.stack(slices)
        else:
            contents = full_detector_response._file['DRM']['CONTENTS']

        self._image_response = Histogram(axes_image_response, contents=contents,
                                         unit = full_detector_response.unit,
                                         dtype = dtype, copy_contents = False)

    def _calc_exposure_map(self):
        """
        Calculate exposure_map, which is an intermidiate matrix used in RL algorithm.
        """

        logger.info("Calculating an exposure map...")

        if self._coordsys_conv_matrix is None:
            exposure_map  = np.sum(self._image_response.contents, axis = (2,3,4))
            # [NuLambda(lb), Ei, Em, Phi, PsiChi] -> [NuLambda(lb), Ei]
        else:
            exposure_map = tensordot_sparse(np.sum(self._coordsys_conv_matrix, axis = (0)),
                                            self._coordsys_conv_matrix.unit,
                                            np.sum(self._image_response, axis = (2,3,4)),
                                            axes = (1, 0))
            # [Time/ScAtt, lb, NuLambda] -> [lb, NuLambda]
            # [NuLambda, Ei, Em, Phi, PsiChi] -> [NuLambda, Ei]
            # [lb, NuLambda] x [NuLambda, Ei] -> [lb, Ei]

        exposure_map *= self.model_axes['lb'].pixarea()

        self._exposure_map = Histogram(self._model_axes, contents = exposure_map, copy_contents = False)

        logger.info("Finished...")

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def exmath(ex, bg, bgnorm):
        for i in prange(len(ex)):
            ex[i] += bg[i] * bgnorm

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def acc_mul(ex, v):
        for i in prange(len(ex)):
            ex[i] = ex[i] * v

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def acc_muladd(ex, v, v0):
        for i in prange(len(ex)):
            ex[i] = ex[i] * v + v0

    def calc_expectation(self, model, dict_bkg_norm = None, almost_zero = 1e-12):
        """
        Calculate expected counts from a given model.

        Parameters
        ----------
        model : :py:class:`cosipy.image_deconvolution.AllSkyImageModel`
            Model map
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}
        almost_zero : float, default 1e-12
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Expected count histogram

        Notes
        -----
        This method should be implemented in a more general class, for example, extended source response class in the future.
        """

        # Currently (2024-01-12) this method can work for both local coordinate CDS and in galactic coordinate CDS.
        # This is just because in DC2 the rotate response for galactic coordinate CDS does not have an axis for time/scatt binning.
        # However it is likely that it will have such an axis in the future in order to consider background variability depending on time and pointing direction etc.
        # Then, the implementation here will not work. Thus, keep in mind that we need to modify it once the response format is fixed.

        if self._coordsys_conv_matrix is None:
            expectation = tensordot_resp( model.contents, self._image_response, axes = ((0,1),(0,1)))
            # ['lb', 'Ei'] x [NuLambda(lb), Ei, Em, Phi, PsiChi] -> [Em, Phi, PsiChi]
        else:
            map_rotated = tensordot_sparse(self._coordsys_conv_matrix.contents,
                                           self._coordsys_conv_matrix.unit,
                                           model.contents,
                                           axes = (1, 0))
            # ['Time/ScAtt', 'lb', 'NuLambda'] x ['lb', 'Ei'] -> [Time/ScAtt, NuLambda, Ei]
            # the unit of map_rotated is 1/cm2 ( = s * 1/cm2/s/sr * sr)

            expectation = tensordot_resp(map_rotated, self._image_response, axes = ((1,2), (0,1)))
            # [Time/ScAtt, NuLambda, Ei] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, Em, Phi, PsiChi]

        ex = expectation.ravel()

        # expectation *= model.axes['lb'].pixarea()
        # expectation += almost_zero
        self.acc_muladd(ex, model.axes['lb'].pixarea().value, almost_zero)

        if dict_bkg_norm is not None:
            for key in self.keys_bkg_models():
                #expectation += self.bkg_model(key).contents * dict_bkg_norm[key]
                self.exmath(ex, self.bkg_model(key).contents.ravel(), dict_bkg_norm[key])

        # patch unit of result to reflect multiply by pixarea
        return Histogram(self.data_axes, contents = expectation.value, unit = expectation.unit * u.sr,
                          copy_contents = False)

    def calc_T_product(self, dataspace_histogram):
        """
        Calculate the product of the input histogram with the transonse matrix of the response function.
        Let R_{ij}, H_{i} be the response matrix and dataspace_histogram, respectively.
        Note that i is the index for the data space, and j is for the model space.
        In this method, \sum_{j} H{i} R_{ij}, namely, R^{T} H is calculated.

        Parameters
        ----------
        dataspace_histogram: :py:class:`histpy.Histogram`
            Its axes must be the same as self.data_axes

        Returns
        -------
        :py:class:`histpy.Histogram`
            The product with self.model_axes
        """
        # TODO: currently, dataspace_histogram is assumed to be dense.

        if self._coordsys_conv_matrix is None:
            tprod = tensordot_resp(dataspace_histogram.contents, self._image_response, axes = ((0,1,2), (2,3,4)))
            # [Em, Phi, PsiChi] x [NuLambda (lb), Ei, Em, Phi, PsiChi] -> [NuLambda (lb), Ei]
        else:
            _ = tensordot_resp(dataspace_histogram.contents, self._image_response, axes = ((1,2,3), (2,3,4)))
            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]

            tprod = tensordot_sparse(self._coordsys_conv_matrix.contents,
                                     self._coordsys_conv_matrix.unit,
                                     _,
                                     axes = ((0,2), (0,1)))
             # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]


        tp = tprod.ravel()

        #tprod *= self.model_axes['lb'].pixarea()
        self.acc_mul(tp, self.model_axes['lb'].pixarea().value)

        # patch unit of result to reflect multiply by pixarea
        return Histogram(self.model_axes, contents = tprod.value, unit = tprod.unit * u.sr,
                         copy_contents = False)

    def calc_bkg_model_product(self, key, dataspace_histogram):
        """
        Calculate the product of the input histogram with the background model.
        Let B_{i}, H_{i} be the background model and dataspace_histogram, respectively.
        In this method, \sum_{i} B_{i} H_{i} is calculated.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram: :py:class:`histpy.Histogram`
            its axes must be the same as self.data_axes

        Returns
        -------
        float
        """
        # TODO: currently, dataspace_histogram is assumed to be a dense.
        return cp.dot(cp.array(dataspace_histogram.contents.ravel()),
                      self.bkg_model(key).cu_array).item()


    def calc_loglikelihood(self, expectation):
        """
        Calculate log-likelihood from given expected counts or model/expectation.

        Parameters
        ----------
        expectation : :py:class:`histpy.Histogram`
            Expected count histogram.

        Returns
        -------
        float
            Log-likelihood
        """
        ev = self.event.contents
        ex = expectation.contents

        loglikelihood = self.ll(ev.ravel(), ex.ravel())
        #loglikelihood = np.sum( ev * np.log(ex) ) - np.sum(ex)

        return loglikelihood

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def ll(ev, ex):

        llhood = 0.

        for i in prange(len(ev)):
            llhood += ev[i] * np.log(ex[i])

        for i in prange(len(ev)):
            llhood -= ex[i]

        return llhood
