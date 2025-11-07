#! /bin/sh
# ---------------------------------------------
# Input:  ./images/splash/*
# Output: ./images/splash_with_version.webp
# ---------------------------------------------

# --- check imagemagick version ---
echo "------ ImageMagick version info --------------------------------------------"
magick identify -version
echo "----------------------------------------------------------------------------"

# --- argument handling ---
DISPLAY_VERSION="$1"  # e.g. "v0.1.10" or "v0.1.11-dev"

# --- create splash without version info ---
if [ ! -f ./images/splash/_splash_without_version.png ]; then
  magick -pointsize 96 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/splash/splash.webp" -gravity South -fill "#aaaaaa" -annotate +0+50 "Collection of Python utilities for personal use." "./images/temp.mpc"
  magick -pointsize 36 -font "./images/splash/google_fonts_montserrat_italic.ttf" "./images/temp.mpc" -gravity SouthWest -fill "#aaaaaa" -annotate +10+5 "DiffusionBee 2.5.3 (FLUX.1-dev + Real-ESRGAN)" "./images/_splash_without_version.png"
fi

# --- add version info ---
magick -pointsize 128 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/_splash_without_version.png" -gravity West -fill "black" -annotate +1403+563 "${DISPLAY_VERSION}" "./images/temp.mpc"
magick -pointsize 128 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/temp.mpc" -gravity West -fill "white" -annotate +1400+560 "${DISPLAY_VERSION}" -quality 100 -define webp:lossless=true "./images/splash_with_version.webp"

# --- clean up ---
echo "Cleaning up..."
rm ./images/*.mpc
rm ./images/*.cache




