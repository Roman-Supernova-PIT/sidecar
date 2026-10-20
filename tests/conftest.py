def pytest_addoption(parser):
    group = parser.getgroup("sky_subtract_and_detect diagnostics")
    group.addoption("--roman-image", default=None, help="Roman cal ASDF file for test_sky_subtract_and_detect")
    group.addoption("--star-catalog", default=None, help="Pre-retrieved star catalog (ecsv) to use instead of Gaia")
    group.addoption("--gaia-mag-limit", type=float, default=17.0, help="Reject sources near Gaia stars brighter than this")
    group.addoption("--gaia-match-radius", type=float, default=5.0, help="Gaia match radius in pixels")
    for name in ("max-peak", "size-clip-sigma", "max-ellipticity", "min-separation", "edge-buffer", "min-bad-distance"):
        group.addoption(f"--{name}", type=float, default=None, help=f"Source selection cut: {name}")
    group.addoption("--bright-star-mag", type=float, default=None, help="Mask halos around Gaia stars brighter than this")
    group.addoption("--bright-star-radius", type=float, default=30.0, help="Halo radius (px) at --bright-star-mag")
    group.addoption("--bright-star-scale", type=float, default=0.2, help="Halo radius growth with magnitude")
    group.addoption("--bright-star-max-radius", type=float, default=None, help="Cap on halo radius (px)")
