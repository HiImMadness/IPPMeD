Data need to be put in a folder next to the IPPMed one, with train and test files in it. 

First you must update the varibale path line = 100 in swinunetr_2gpu.py and in test.py.
Next you need to run swinunetr_2gpu.py. It saves a model every 20 epoch, then check on wandb to find the best model using the score metrics, it should be around epoch = 600 dont go too high to avoid overfitting.

Then you can predict using this model in test.py you just need to update the model path line 366. It'll save the predicted images in the test folder in IPPMed.