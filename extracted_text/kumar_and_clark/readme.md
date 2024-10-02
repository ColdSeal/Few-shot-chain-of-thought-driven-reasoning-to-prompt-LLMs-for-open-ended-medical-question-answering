### Extraction details
- Used the training script *data_extract.py* to extract out the data from pdf file.
- All files contain the **respiratory** and **endocrine** systems
- Files included {name} = "kumar_and_clark":
    - {name}.pdf : Text containing all font sizes
    - {name}\_top\_{k}.pdf : Text with only the top {k} font sizes are being considered
    - The best one seems to be top_3 since it contains most of the information, and reducing the {k} value takes away some key information while only reducing special characters like {bullets, commas, titles}, which isn't very useful.