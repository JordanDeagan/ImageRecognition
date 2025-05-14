# Import google_streetview for the api module
import google_streetview.api
import csv


with open("input.csv", "r") as f:
    reader = csv.reader(f, delimiter=',')
    for i, line in enumerate(reader):
        print(line[0]+ ' ' + line[1])
        print(i)
        params = [{
            'size': '640x640',
            'location':line[1]+','+line[0],
            'heading': '210',
            'pitch': '0',
            'key': 'AIzaSyCKRTI6u1HCzzTN7VQAdE7O3HD-Hkr1zBE'
        }]

        # Create a results object
        results = google_streetview.api.results(params)

        # Download images to directory 'downloads'
        results.download_links('downloads_'+str(i))
