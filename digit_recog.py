
from neuron import Neuron
from PIL import Image
import os
import sys
import pickle

def main():
  
  # neuron attributes
  threshold  = 1
  width      = 35
  l_rate     = 0.005
  err_margin = 0.1  # does nothing so far
  a_func     = "step"
  f_stretch  = 1
  
  # create list of neurons, one to be trained for each number
  neurons = list()
  
  for x in range(10):
    neurons.append(Neuron(width, a_func, f_stretch, threshold, l_rate, err_margin))
  
  # train or load from file if save file exists
  train(neurons)
  
  # trial
  test_img = Image.open(sys.argv[1])
  
  for x in range(len(neurons)):
    n = neurons[x]
    feed(test_img, n)
    n.activate()
    if n.get_output() == 1.0:
      print "NEURON %i IS RESPONDING" %x


def train(neurons):
  """
    Use data from training-sets directory to train.
  """
  for digit in range(len(neurons)):
    
    print "TRAINING FOR %i" %digit
    
    n = neurons[digit]
    
    ls1 = os.listdir("training-sets/")
    ls1 = sorted(ls1)
    
    
    errors  = 1
    counter = 0
    
    while errors > 0:
      errors = 0
      for i in ls1: # for every directory in training sets
        
        ### SET EXPECTED OUTPUT FOR CURRENT DIRECTORY
        try:
          exp_out = 0   # for other digits, training as not recognized (0)
          if int(i) == digit:
            exp_out = 1 # corresponding digit,  training as recognized (1)
        except ValueError:
          break # ignore directories that are not named as an integer
        
        ### FOR EACH IMAGE, FEED PIXEL SET AS INPUT AND USE exp_out TO CHECK
        dir1 = "training-sets/%s/" %i
        ls2 = os.listdir(dir1)
        ls2 = sorted(ls2)
        for j in ls2: # for every image
          
          img = Image.open(dir1 + j)
          feed(img, n)
          counter += 1
          errors += n.train_step(exp_out) # train with inputs
        # end of j-loop
      # end of i-loop
      print "Errors: %i" %errors
    # end of while loop
    
    
    
    print "Images processed: %i" %counter
    
  # end of for loop
  out = ""
  for x in range(len(neurons)):
    n = neurons[x]
    out += str(x) + " " # insert digit
    for i in range(n.get_width()): # insert weights
      out += str(n.get_weight(i)) + " "
    out += str(n.get_threshold()) + "\n"
    
  f = open("train-saves/train.save", "w")
  f.write(out)
  f.close()
# end of def train
  

def feed(img, n):
  """
    Sets the inputs of neuron according to pixel data provided of image
    PARAMS:
      img : Image to take pixel data from
      n   : neuron
  """
  pixels = img.load()
  width  = img.size[0]
  height = img.size[1]
  # distill pixel values into input nodes
  for w in range(width):
    for h in range(height):
      n.set_input(w * height + h, pixels[w,h][0]) # takes just the red value



if __name__ == "__main__":
  try:
    test_img = Image.open(sys.argv[1])
  except IOError:
    print "Error: Image file cannot be read."
    print "Usage: python digit_recog.py path/to/img.png"
    exit(1)
  except IndexError:
    print "Error: Argument not found."
    print "Usage: python digit_recog.py path/to/img.png"
    exit(1)
  
  main()
