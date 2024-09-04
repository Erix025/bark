from utils import submission_test
import sys
from utils import submission_test

# labels_file = "data/labels_test_finetuned.csv"
if len(sys.argv) != 3:
    print("Usage: python check.py <labels_file> <submission_file>")
    sys.exit(1)

labels_file = sys.argv[1]
submission_file = sys.argv[2]

score = submission_test.submission_loss(labels_file, submission_file)
print("Score:", score)