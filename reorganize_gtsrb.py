import os
import pandas as pd
import shutil

# Read the CSV file
csv_path = '.datasets/gtsrb/GTSRB/Final_Test/Images/GT-final_test.test.csv'
df = pd.read_csv(csv_path, sep=';')

# Create a backup of the original files
test_dir = '.datasets/gtsrb/GTSRB/Final_Test/Images'
backup_dir = '.datasets/gtsrb/GTSRB/Final_Test/Images_backup'

# Create backup directory if it doesn't exist
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# Move all .ppm files to backup
for file in os.listdir(test_dir):
    if file.endswith('.ppm'):
        src = os.path.join(test_dir, file)
        dst = os.path.join(backup_dir, file)
        shutil.move(src, dst)

# Now rename and move files back according to CSV
for index, row in df.iterrows():
    old_name = str(index + 11631) + '.ppm'  # Current naming pattern
    new_name = row['Filename']  # Target naming pattern from CSV
    
    src = os.path.join(backup_dir, old_name)
    dst = os.path.join(test_dir, new_name)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)  # Use copy2 to preserve metadata

print("Files have been reorganized according to the CSV file.") 