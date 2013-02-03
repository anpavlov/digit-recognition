
# 5x7 Digit Image Recognition: Python

## How to run

1. \[Optional: I've already done this\] Run digit_maker.py in your terminal. This will generate noisy renditions of the clean PNGs in `training-sets/`.

        python digit_maker.py
        
2. Run digit_recog.py. As the argument, add the location of your 5x7px PNG file. For example, if I have a PNG file named `0_25.png` located in `training-sets/0/`, I would type:

        python digit_recog.py training-sets/0/0_25.png
    
3. From there, `digit_recog.py` will indicate the number it thinks you have shown it.

NB: Every time you run `digit_recog.py`, it will train using the files in `training-set`. This takes approximately 5 seconds for me. I have not yet implemented a Neuron saving feature... but you can.

## The theory

I'm using the following weight adjustment formulae:

    delta weight = learning_rate * error * input
    delta threshold = -(learning_rate * error)

And the activation function is a step function, 0 being false and 1 being true. You can try changing the Neuron parameter called `a_func` in `digit_recog.py` to `"sigmoid"` to get a sigmoidal activation function, but I have not tested this yet, and there may be other features I have not researched about using sigmoidal activation functions.

## The algorithm

1. There are 10 Neurons, each to be trained to its own digit (0-9).
2. There is 1 clean image of each number. Thirty noisy renditions of each number are generated. These 310 images constitute the training set.
3. Each neuron is fed 310 images (31 images per number for 10 numbers) and told whether to recognize it or not. For example, the neuron that is designated to recognize the number 1 will be told not to recognize 0, 2, 3, ... 9, and to recognize 1. It will loop back to the beginning of the training set until it stops making mistakes.
4. After training, the neuron should be fed an image that it has not seen before (simply generate a new noisy image) to test if the neuron has been properly trained.

---

There are a multitude of algorithms I could have used to hardcode number recognition at this low resolution, and probably in a simpler fashion as well. I chose to use a neural network so that I could familiarize myself with the theory and the algorithms before scaling it up to multiple layers and multiple neurons per layer for problems of higher complexity. My Neuron class is written with this thought in mind.

Feel free to use any part of this repository for your own projects. Although the Neuron class is not comprehensive (as my artificial intelligence training is so far limited), I've put a lot of effort into making it versatile for the different usage scenarios that I am aware of.
