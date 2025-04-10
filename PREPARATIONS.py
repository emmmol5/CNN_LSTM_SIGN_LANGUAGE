from IMPORTS import *

######## Lim inn andre endringer du har gjort med metadataen

#df = pd.read_excel(r'C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\helse_ordliste.xlsx')

df = pd.read_excel(org_metadata)

# Rename columns 
df.columns = ['Health_Term', 'Video_File']

# Drop the first row 
df = df.iloc[1:].reset_index(drop=True)

# Function to remove trailing numbers from the "Health_Term" index
'''def clean_health_term(df: pd.DataFrame):
    df['Health_Term'] = df['Health_Term'].str.replace(r'\s\d+$', '', regex=True).str.strip()'''

# Function replace invalid characters for windows folder names
def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  
    filename = re.sub(r'[.-]+$', '', filename)
    return filename.strip()

# Apply cleaning functions
#clean_health_term(df)
df['Health_Term'] = df['Health_Term'].apply(sanitize_filename)

# Save the modified file 
df.to_excel(r'C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\helse_ordliste_mod.xlsx', index=False)


'''# Function to create the "Extracted_frames" column by removing ".mp4" from "Video_File"
def add_column(df: pd.DataFrame):
    df["Extracted_frames"] = df["Video_File"].str.replace(r'\.mp4$', '', regex=True)'''

# Read the Excel file
#health_term = pd.read_excel(metadata, index_col="Health_Term")

# Modify the dataset
#modify_health_term(health_term)
'''add_column(health_term)'''

# Save the modified file
#health_term.to_excel(metadata)
