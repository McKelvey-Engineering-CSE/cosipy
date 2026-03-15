import sys
import pandas as pd
import numpy as np
import healpy as hp

import matplotlib
import matplotlib.pyplot as plt

def plot_healpix_err(csv_file, save_file, nside):

    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Filter upper hemisphere
    df = df[df['alt'] >= 0]

    # 3. Convert to HEALPix pixels
    theta = np.radians(90.0 - df['alt'].values)
    phi = np.radians(df['az'].values)

    df['pixel'] = hp.ang2pix(nside, theta, phi)

    agg_df = df.groupby('pixel')['err'].agg(
        median='median',
        p68=lambda x: x.quantile(0.68),
        p95=lambda x: x.quantile(0.95)
    ).reset_index()

    # 4. Create maps
    npix = hp.nside2npix(nside)

    hpx_map_med = np.full(npix, hp.UNSEEN)
    hpx_map_68 = np.full(npix, hp.UNSEEN)
    hpx_map_95 = np.full(npix, hp.UNSEEN)

    hpx_map_med[agg_df['pixel']] = agg_df['median'].values
    hpx_map_68[agg_df['pixel']] = agg_df['p68'].values
    hpx_map_95[agg_df['pixel']] = agg_df['p95'].values

    # ------------------------------------------------
    # Label function (safe version)
    # ------------------------------------------------

    brightness_weights = np.array([0.299, 0.587, 0.114])

    def annotate_map_centers(hpx_map, nside_val):

        valid_pixels = np.where(hpx_map != hp.UNSEEN)[0]

        lon_centers, lat_centers = hp.pix2ang(nside_val, valid_pixels, lonlat=True)
        values = hpx_map[valid_pixels]

        # force labels close to edge of plot all the way to the edge
        lat_centers[lat_centers < 10] = 0.

        proj = hp.projector.OrthographicProj(rot=(0, 90, 0), half_sky=True)
        x, y = proj.ang2xy(lon_centers, lat_centers, lonlat=True)

        # for centers close to edge of plot, nudge the text
        # coordinates just past the edge of the plot. We have to do it
        # as follows because ang2xy() doesn't work for latitudes < 0.

        # compute "outward" unit vectors from center of plot
        xp, yp = proj.ang2xy(0., 90., lonlat=True)
        dx, dy = x - xp, y - yp
        dx /= np.sqrt(dx**2 + dy**2)
        dy /= np.sqrt(dx**2 + dy**2)

        # a nudge of 0.06 looks visusally nice
        x[lat_centers == 0] += 0.06 * dx[lat_centers == 0]
        y[lat_centers == 0] += 0.06 * dy[lat_centers == 0]

        # create a mapping from plotted values to colors in our color map
        cmap = plt.get_cmap()
        norm = matplotlib.colors.Normalize(vmin=np.min(values),
                                           vmax=np.max(values))
        color_mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        for xi, yi, lati, val in zip(x, y, lat_centers, values):


            if lati == 0:
                # label is outside boundary of plot
                textcolor = "black"
            else:
                # choose a contrasting color for the label on each patch
                rgba_color = np.array(color_mapper.to_rgba(val)[:3])
                brightness = np.dot(rgba_color, brightness_weights)
                textcolor = "white" if brightness <= 0.500 else "black"

            txt = plt.text(
                xi,
                yi,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color=textcolor,
                fontweight="bold"
            )

    # ------------------------------------------------

    plt.clf()
    plt.figure(figsize=(24, 8))

    # Median
    hp.orthview(
        hpx_map_med,
        half_sky=True,
        title="",
        unit="median error (deg)",
        cmap="viridis",
        rot=(0, 90, 0),
        sub=131
    )

    hp.graticule(dpar=15, dmer=30)
    annotate_map_centers(hpx_map_med, nside)

    # 68%
    hp.orthview(
        hpx_map_68,
        half_sky=True,
        title="",
        unit="68% containment error (deg)",
        cmap="viridis",
        rot=(0, 90, 0),
        sub=132
    )

    hp.graticule(dpar=15, dmer=30)
    annotate_map_centers(hpx_map_68, nside)

    # 95%
    hp.orthview(
        hpx_map_95,
        half_sky=True,
        title="",
        unit="95% containment error (deg)",
        cmap="viridis",
        rot=(0, 90, 0),
        sub=133
    )

    hp.graticule(dpar=15, dmer=30)
    annotate_map_centers(hpx_map_95, nside)

    plt.savefig(save_file, bbox_inches="tight", dpi=150)
    plt.show()


if __name__ == "__main__":

    infile = sys.argv[1]
    outfile = sys.argv[2]

    nside = 4

    plot_healpix_err(infile, outfile, nside)
