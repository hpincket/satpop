import bs4
import time
import os
import random
import csv
import requests
import constants as C
import json
from uuid import uuid4

# Init Gaz Index
gaz_index = {}
with open(C.GAZ_FILE, "r", encoding="utf-8", newline='') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
    for row in tsvreader:
        gaz_index[row[1]] = row

print(len(gaz_index.keys()))

def get_pop(lat, lon):
    r = requests.get(C.CENSUS_URL, {"latitude": lat,
                                    "longitude": lon,
                                    "show_all": True,
                                    "format": "json"})
    res_json = json.loads(r.content.decode('utf-8'))
    if res_json['status'] != "OK":
        print(res_json)
        return None
    try:
        FIPS = res_json['Block']['FIPS'][:-4]
    except TypeError:
        print(res_json)
        return None
    if FIPS in gaz_index:
        tract = gaz_index[FIPS]
        try:
            return int(tract[2]) / (int(tract[4]) / 1000000)
        except ZeroDivisionError:
            return 0
    else:
        print("{} not found in index".format(FIPS))



def get_image(given_uuid, lat, lon):
    r = requests.get(C.NASA_URL, {"api_key": C.NASA_API_KEY,
                                  # "date": C.DATE,
                                  "cloud_score": True,
                                  "lat": str(lat),
                                  "lon": str(lon)})

    if r.status_code == 429:
        print("Rate Limited.")
        print(r.content)
        if 'Retry-After' in r.headers:
            print("Sleepy time: {}".format(r.headers['Retry-After']))
            time.sleep(int(r.headers['Retry-After']))
        else:
            time.sleep(300)
        return False
    if r.status_code != 200:
        print(given_uuid, lat, lon, r.status_code)
        return False
    res_json = json.loads(r.content.decode('utf-8'))
    cloud_score = 1
    try:
        url = res_json["url"]
        cloud_score = res_json["cloud_score"]
    except:
        print(res_json)
        print("Error - ({},{})".format(lat, lon))
    if cloud_score == None or cloud_score > .5:
        return False
    r = requests.get(url)
    fname = os.path.join(C.SATPOP_IMAGE_FOLDER, "{}.{}".format(given_uuid, "png"))
    with open(fname, "+wb") as fd:
        fd.write(r.content)
    return True


class GeoNamesData:
    """
    The main 'geoname' table has the following fields :
    ---------------------------------------------------
    geonameid         : integer id of record in geonames database
    name              : name of geographical point (utf8) varchar(200)
    asciiname         : name of geographical point in plain ascii characters, varchar(200)
    alternatenames    : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
    latitude          : latitude in decimal degrees (wgs84)
    longitude         : longitude in decimal degrees (wgs84)
    feature class     : see http://www.geonames.org/export/codes.html, char(1)
    feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
    country code      : ISO-3166 2-letter country code, 2 characters
    cc2               : alternate country codes, comma separated, ISO-3166 2-letter country code, 200 characters
    admin1 code       : fipscode (subject to change to iso code), see exceptions below, see file admin1Codes.txt for display names of this code; varchar(20)
    admin2 code       : code for the second administrative division, a county in the US, see file admin2Codes.txt; varchar(80)
    admin3 code       : code for third level administrative division, varchar(20)
    admin4 code       : code for fourth level administrative division, varchar(20)
    population        : bigint (8 byte int)
    elevation         : in meters, integer
    dem               : digital elevation model, srtm3 or gtopo30, average elevation of 3''x3'' (ca 90mx90m) or 30''x30'' (ca 900mx900m) area in meters, integer. srtm processed by cgiar/ciat.
    timezone          : the iana timezone id (see file timeZone.txt) varchar(40)
    modification date : date of last modification in yyyy-MM-dd format
    """

    def _write_data_cache(self):
        with open(self.cache_path, '+w', encoding="utf-8", newline='') as tsvfile:
            tsvwriter = csv.writer(tsvfile, delimiter='\t',
               quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for e in self.data:
                tsvwriter.writerow(e)

    def _read_data_cache(self):
        read_data = []
        with open(self.cache_path, "r", encoding="utf-8", newline='') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t',
                                   quotechar='|')
            for row in tsvreader:
                read_data.append(row)
        self.data = read_data

    def _gen_new_data(self):
        all_data = []
        with open(self.path, 'r', encoding='utf-8', newline='') as csvfile:
            geonames_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in geonames_reader:
                if row[6] == "A":
                    all_data.append(row)
        size = min(len(all_data), C.NUM_GEO_CITIES)
        self.data = random.sample(all_data, size)

    def __init__(self, path, cache_path):
        self.path = path
        self.cache_path = cache_path
        self.pos = 0
        if os.path.exists(cache_path):
            self._read_data_cache()
        else:
            self._gen_new_data()
            self._write_data_cache()
        print("Init done")

def image_already_downloaded(given_uuid):
    fname = os.path.join(C.SATPOP_IMAGE_FOLDER, "{}.{}".format(given_uuid, "png"))
    return os.path.exists(fname)

def main():
    gnd = GeoNamesData(C.SATPOP_GEONAMES_FILE, C.SATPOP_GEONAMES_CACHE_FILE)
    main_data_fille = os.path.join(C.SATPOP_DATA_FOLDER, "data.tsv")
    # Cache -> Data
    to_skip = 0
    if os.path.exists(main_data_fille):
        with open(main_data_fille, "r", newline='') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
            for i,row in enumerate(tsvreader):
                to_skip = i+1
    print("Will skip {} data rows.".format(to_skip))
    with open(main_data_fille, "a", newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t', quotechar='|')
        for i,data in enumerate(gnd.data):
            if i < to_skip:
                continue
            if i % 100 == 0:
                tsvfile.flush()
            current_uuid = uuid4()
            lat, lon = data[4], data[5]
            pop = get_pop(lat, lon)
            if pop is not None:
                tsvwriter.writerow([current_uuid, lat, lon, pop])
    print('Images, get."')
    # Get Images
    with open(main_data_fille, "r", newline='') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        for row in tsvreader:
            get_image(row[0], row[1], row[2])

if __name__ == "__main__":
    main()

