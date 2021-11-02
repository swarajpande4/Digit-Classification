import requests 

print("Fetching train_data...")
train_req = requests.get("https://pjreddie.com/media/files/mnist_train.csv")
csv_file = open("data/train_data.csv", "wb")
csv_file.write(train_req.content)
csv_file.close()
print("done.")

print("Fetching test_data...")
test_req = requests.get("https://pjreddie.com/media/files/mnist_test.csv")
csv_file = open("data/test_data.csv", "wb")
csv_file.write(test_req.content)
csv_file.close()
print("done.")

