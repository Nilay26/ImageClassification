exec(compile(open('importLibraries.py', "rb").read(), 'importLibraries.py', 'exec'))


arg_parser = argparse.ArgumentParser()

training_batch, validation_batch, test_batch, training_data = transform_data.image_augmentation()
#Printing to check if data loaded properly

img_iterator = iter(training_batch)
images, labels = img_iterator.next()
print(type(images))
print(images.shape)
print(labels.shape)

arg_parser.add_argument('--architecture', dest="architecture", action="store", default="vgg16", type = str)
arg_parser.add_argument('--hiddenlayer_units', dest="hiddenlayer_units", action="store", default = 4096)
arg_parser.add_argument('--learning_rate', dest="learning_rate", action="store", default = 0.0001)
arg_parser.add_argument('--device', dest="device", action="store", default = 'cuda' , type = str)

model, criterion, optimizer = build_Classifier.build_layer(arg_parser.parse_args().architecture, arg_parser.parse_args().hiddenlayer_units, arg_parser.parse_args().learning_rate, arg_parser.parse_args().device)

print(model)
arg_parser.add_argument('--epochs', dest="epochs", action="store", default = 10 , type = int)
train_data.train_data_nn(arg_parser.parse_args().device,training_batch, arg_parser.parse_args().epochs, model, optimizer, criterion, validation_batch)

#Save checkpoint function call
arg_parser.add_argument('--model_path', dest='model_path', action = 'store', default='./checkpoint.pth')
train_data.save_model(model, arg_parser.parse_args().architecture, arg_parser.parse_args().hiddenlayer_units, arg_parser.parse_args().learning_rate, arg_parser.parse_args().model_path, training_data)
