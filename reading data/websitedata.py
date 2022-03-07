import requests
import csv
from bs4 import BeautifulSoup
# Make a request
page = requests.get(
    "https://www.bue.edu.eg/tuition_annual_fees/#For-Egyptian-Students")
soup = BeautifulSoup(page.content, 'html.parser')

# Create all_h1_tags as empty list
all_h1_tags = []
test = []
f = open('test.csv','w',encoding='UTF8')
writer = csv.writer(f)

# Set all_h1_tags to all h1 tags of the soup
for element in soup.select('tr'):
    #if ("Arts & Design" in element.text):
    all_h1_tags.append(element.text)
   

    for elementt in element.select('td'):
        test.append(elementt.text)
    writer.writerow(test)
    test.clear()
        
    

# Create seventh_p_text and set it to 7th p element text of the page
seventh_p_text = soup.select('p')[6].text

f.close()
print(all_h1_tags[6])