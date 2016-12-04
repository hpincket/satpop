import os

DATE="2010-05-05"
# NASA_ACCOUNT_ID = "0986875a-5914-4bc2-a16e-d28f462abac3"
# NASA_API_KEY = "B7cSnefXt69xFOz8SLqWtdeSYpYmTIiwYN1fsQbs"
NASA_ACCOUNT_ID = "ca95d67c-25da-454c-bf73-c4731fe70ffd"
NASA_API_KEY = "E0YgiItZEuN3DHJsKuxnUZmHV1FVlrqjOOI0NHTA"
NASA_URL = "https://api.nasa.gov/planetary/earth/imagery"

CENSUS_URL = "http://data.fcc.gov/api/block/find" # ?format=json&latitude=42.456&longitude=-74.987&showall=true"

SATPOP_DATA_FOLDER = os.environ.get('SATPOP_DATA_FOLDER', r'SATPOP_DATA')
SATPOP_IMAGE_FOLDER = os.path.join(SATPOP_DATA_FOLDER, os.environ.get('SATPOP_IMG_FOLDER', 'sat_images'))
SATPOP_GEONAMES_FILE = os.path.join(SATPOP_DATA_FOLDER, 'US.txt')
SATPOP_GEONAMES_CACHE_FILE = os.path.join(SATPOP_DATA_FOLDER, 'US.cache')

SATPOP_ERR_IMAGE_FILE = os.path.join(SATPOP_DATA_FOLDER, "image-errors.txt")
SATPOP_MAIN_DATA_FILE = os.path.join(SATPOP_DATA_FOLDER, "data.tsv")

GAZ_FILE = os.path.join(SATPOP_DATA_FOLDER, "Gaz_tracts_national.txt")

NUM_GEO_CITIES = 100000


if not os.path.exists(SATPOP_IMAGE_FOLDER):
    os.makedirs(SATPOP_IMAGE_FOLDER)

