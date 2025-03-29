How It Works:
Reference Faces
Folder : facial_database/
Purpose : Upload images of known individuals (e.g., Jhon.jpg.)


Input Scanning
Folder : input_database/
Purpose : Add images to scan for face recognition.
Process : Detects faces in input images and matches them to the reference database.

Output
Folder : results/
Content : Saved images with bounding boxes and labels (e.g., Alice_1.jpg).


Getting Started:

Add Reference Faces :
Place images of known individuals in facial_database/.
Upload Input Images :
Add images to scan in input_database/.
Run the Program :
The system will:
Detect faces in input images.
Compare embeddings to the reference database.
Save results to results/.
