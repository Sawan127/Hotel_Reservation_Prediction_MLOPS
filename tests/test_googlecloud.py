from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket("sawan_hotelreservation_bucket_0866")
blobs = bucket.list_blobs()

for blob in blobs:
    print(blob.name)  # Make sure your target file is listed here

