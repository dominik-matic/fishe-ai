# Adriatic Fish Classifier

This is an old project where I didn't use my convenient ml-templates and is something I'm hoping to implement in the near future. This project consists of a dataset scraped from Google Images of over 400 species of fish found in the Adriatic Sea. I'm testing out a lot of image classifier models, some scrambled together by me and some pretrained. In the end, I think it'll come down to building some sort of ensemble classifier that will yield the best results.

Working on this project is somewhat slow because training any particular model can take upwards of a week in real time. If I manage to get my hands on more and better hardware I might work on this a bit more.

So far, the best possible accuracy was in the ballpark of 40%. I suspect a couple of reasons for this:
- still haven't found a suitable model for the task
- the dataset is simply bad, there are 442 species and each of them can have between 1-50 images which may not be enough and a lot of them could be misclassified
- a lot of species look a lot like, if not identical, to other, closely related, species from the dataset

Because of this, there are a couple of things I'd like to do:

- experiment with more architectures and ensembles
- try boosting
- analyze the confusion matrix to deduce which species can be grouped together in a single class
