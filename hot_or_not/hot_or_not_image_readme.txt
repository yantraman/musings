hot_or_not_image_and_rating_data.zip is a zip file containing the images taken from hotornot.com that were used in our Hot or Not experiments.

It has two directories:
	female/
	male/

female/ and male/ contain 1000 female images and 1000 male images respectively (gender reported on hotornot.com was assumed to be accurate) {0001..1000}.jpg. 

Each image has a corresponding text file, {0001..1000}.txt, which contains 5 parts.  For example, here is the content of female/0340.txt:
	8.1
	2687
	http://www.hotornot.com/r/?eid=SYNEAZE-NDH
	http://p1.hotornot.com/pics/HE/HY/KZ/KM/SYNEAZEDBFFX.jpg
	hi, i'm 29 years old, sensable, funny and reliable. enjoys horse riding, music, films, being in good relaxing company.

The first line (8.1) is the image's rating.  
The second line (2687) is the number of users who have rated it.  
The third line (http://www.hotornot.com/r/?eid=SYNEAZE-NDH) is the URL of the "rating page" where users can rate the image.
The fourth line (http://p1.hotornot.com/pics/HE/HY/KZ/KM/SYNEAZEDBFFX.jpg) is the direct link to the image.
The remainder of the file (hi, i'm 29 years old, ...) can be 0 or more lines, and contains the user's self-description that is displayed below his/her image.  This was not used in our experiments.

Finally, the female/ and male/ directories each contain a .mat data file (female.mat and male.mat) which contains a single 1000x3 matrix.  Each row corresponds to an image.  

The first column is the image number in the directory.
The second column is the image's rating.
The third column is the number of raters.

If you only want the ratings for each image, you can use the .mat files instead of parsing the full .txt files for each image.
