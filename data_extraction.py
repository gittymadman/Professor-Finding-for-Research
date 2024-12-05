from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

url = "https://www.iiitdm.ac.in/people/faculty"
page = requests.get(url)
soup = BeautifulSoup(page.text,'html.parser')
# print(soup)

# For taking out names of Professors
people = soup.find_all('h1',class_='w-full border-b border-base text-2xl font-medium text-skin-accent')
# name = soup.find('img alt')
Name=[]
for names in people:
    name_text = names.get_text().strip()
    match = re.search(r'(Dr\.|Prof\.|Mr\.\s+)?[A-Za-z.\s]+', name_text)
    Name.append(match[0])

# For taking out their designation
desig = soup.find_all('h2',class_='text-lg font-medium text-accent-secondary')
Designation=[]
for des in desig:
    desig_text = des.get_text().strip()
    match = re.search(r'[A-Za-z\s]+', desig_text)
    Designation.append(match[0])
# print(Designation)

# For taking out their Research Interests
res_int = soup.find_all('p',class_='text-justify')
Research_Interests = []
for res in res_int:
    fin = res.get_text().strip().strip('\n').split('</span>')[0].split('Research Interests:')[1]
    fin=fin.replace('\r\n','')
    Research_Interests.append(fin)
#     res_text = res.get_text().strip()
#     match = re.search(r'[A-Za-z,\s]+', res_text)
#     Research_Interests.append(match[0])
# print(Research_Interests)


# Institute of Study (Ph.D)
wes = soup.find_all('div',class_='flex flex-1 flex-col justify-evenly gap-2')
study = []
for div in wes:
    p_no_class = div.find('p',class_=False)
    if p_no_class:
        s = [p for p in div.find_all('p') if not p.has_attr('class')]
        for res in s:
            res_text = res.get_text().split(' ')
            x=''
            for i in res_text:
                if i !='Ph.D:':
                    x+=str(i)
                    x+=' '
            study.append(x)
    else:
        study.append('null')
# print(study)


# Getting Email id
em = soup.find_all('ul',class_='flex gap-4 border-t border-base pt-2')
Email_id = []
for i in em:
    email = i.find('li')
    email=email.get_text().split(': ')[1]
    Email_id.append(email)
# print(Email_id)

# Getting Faculty location (Cabin No.)
loc = soup.find_all('ul',class_='flex gap-4 border-t border-base pt-2')
Cabin_No=[]
for i in em:
    location = i.find_all('li')
    if len(location)>=2:
        location=location[1].get_text().split(': ')[1:]
        if len(location)>1:
            w=''
            for i in location:
                w+=i
            Cabin_No.append(w)
        else:
            Cabin_No.append(location[0])
    else:
        Cabin_No.append('null')
    
# print(Cabin_No)

Prof_data=pd.DataFrame({
    'Name':Name,
    'Designation':Designation,
    'Institute of Study Ph.D':study,
    'Research Interests':Research_Interests,
    'Email':Email_id,
    'Cabin No':Cabin_No})

Prof_data.to_csv("Professors_extracted_data.csv",index=True,index_label='Index')