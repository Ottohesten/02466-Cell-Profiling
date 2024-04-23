import pandas as pd
data = pd.read_parquet("/Users/rasmusjensen/metadata.parquet") 
classes = data["moa"]
multicellid = data["Multi_Cell_Image_Id"]

# Function to get classes from metadata WORK IN PROGRESS
def get_classes(multicellid, classes):
    class_list = []
    for cell_id, class_label in zip(multicellid, classes):
        class_list.append((cell_id, class_label))
    return class_list


    #Print all moas from get_classes:

for i in get_classes(multicellid, classes):
    print(i[1]) # Print all moas from get_classes
    print(i[0]) # Print all multicellids from get_classes