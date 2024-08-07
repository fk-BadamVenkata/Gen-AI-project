from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import json
import time

url = "https://www.portauthoritynsw.com.au/sydney-harbour/pilotage-navigation/daily-vessel-movements/"

driver = webdriver.Chrome()

driver.get(url)

time.sleep(10)  
table = driver.find_element(By.TAG_NAME, 'table')
headers = [header.text for header in table.find_elements(By.TAG_NAME, 'th')]
rows = []
for row in table.find_elements(By.TAG_NAME, 'tr')[1:]:  
    cells = row.find_elements(By.TAG_NAME, 'td')
    row_data = [cell.text for cell in cells]
    rows.append(row_data)
df = pd.DataFrame(rows, columns=headers)

driver.quit()

df = df.loc[:, ~df.columns.duplicated()]

file_path = 'tabledata.json'

df.to_json(file_path, orient='records', lines=True)

print(f"Table data has been successfully saved to {file_path}")
