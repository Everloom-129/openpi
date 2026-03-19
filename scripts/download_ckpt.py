from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid')
print('Downloaded to:', p)