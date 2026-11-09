import pandas as pd
import os

def load_and_preprocess_data():
    print("Loading dataset sheets...")
    excel_path = "data/raw/online_retail_II.xlsx"
    
    # Read both sheets from the workbook
    df_2009 = pd.read_excel(excel_path, sheet_name='Year 2009-2010')
    df_2010 = pd.read_excel(excel_path, sheet_name='Year 2010-2011')
    
    # Combine both years
    df = pd.concat([df_2009, df_2010], ignore_index=True)
    print(f"Combined raw shape: {df.shape}")
    
    # Basic cleaning
    df = df.dropna(subset=['Customer ID', 'Description'])
    # Remove cancelled orders (Invoice starting with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    # Remove negative/zero quantities or prices
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # Calculate Total Spend per row
    df['TotalSpend'] = df['Quantity'] * df['Price']
    
    # Create a processed output directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save processed dataset
    output_path = "data/processed/cleaned_retail.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    load_and_preprocess_data()