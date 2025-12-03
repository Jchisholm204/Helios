# Testing the Model


To test the model, run the command `python src/evaluate_model.py`.

Before testing, the captions must be encoded using the caption encoder script.
To run the caption encoder script, execute `python src/caption_encoder.py`.
The images must also be encoded using the image encoder script.
To run the image encoder script, execute `python src/image_encoder.py`.

There are four possible models:
- `allmod` - containing the combined model
- `augmentation` - trained using only data agumentation
- `basemodel` - the base model
- `tempsch` - trained with temperature scheduling

To adjust the model being evaluated, both scripts contain a `MODEL` variable.
Change this variable to one of the above names to evaluate it.

