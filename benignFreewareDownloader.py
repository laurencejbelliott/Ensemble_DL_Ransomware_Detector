__author__ = "Laurence Elliott"

from urllib.request import urlopen as uOpen
from bs4 import BeautifulSoup as soup
import re
import os


myUrl = "https://www.freewarefiles.com/search.php?categoryid=1&query=&boolean=exact"

# connecting to and downloading page
uClient = uOpen(myUrl)
page_html = uClient.read()
uClient.close()

# instatiating BeautifulSoup parsing of first page
page_soup = soup(page_html, "html.parser")

# gets page numbers from list above program listings
numPagesA = page_soup.findAll("li",{"class":"page-item"})
numPagesArr = []
for numPageA in numPagesA:
    numPage = numPageA.findAll("a",{"class":"page-link"})[0]
    try:
        numPage = re.search('(?<=>)[0-9]+(?=<\/a>)',str(numPage)).group(0)
        numPagesArr.append(numPage)
    except:
        pass

# the last of the list of page numbers is stored for reference as the last
# page of the search
maxPage = numPagesArr[-1]
print("Total pages: " + str(maxPage) + "\n")

# get next page link
nextA = page_soup.findAll("a",{"aria-label":"Next"})[0]
print(nextA["href"])

# get links to individual program download pages
downloadPageHeaders = page_soup.findAll("h3",{"class":"search-head"})
downloadPageLinks = []
for pageHeader in downloadPageHeaders:
    pageHeader = pageHeader.findAll("a")[0]
    downloadPageLink = pageHeader["href"]
    print(downloadPageLink)
    downloadPageLinks.append(downloadPageLink)

# main
if __name__ == "__main__":
    myUrl = "https://www.freewarefiles.com/search.php?categoryid=1&query=&boolean=exact"
    count = 1

    # connecting to and downloading first page
    uClient = uOpen(myUrl)
    page_html = uClient.read()
    uClient.close()

    # instatiating BeautifulSoup parsing of first page
    page_soup = soup(page_html, "html.parser")

    # gets page numbers from list above program listings
    numPagesA = page_soup.findAll("li", {"class": "page-item"})
    numPagesArr = []
    for numPageA in numPagesA:
        numPage = numPageA.findAll("a", {"class": "page-link"})[0]
        try:
            numPage = re.search('(?<=>)[0-9]+(?=<\/a>)', str(numPage)).group(0)
            numPagesArr.append(numPage)
        except:
            pass

    # the last of the list of page numbers is stored for reference as the last
    # page of the search
    maxPage = int(numPagesArr[-1])
    print("Total pages: " + str(maxPage) + "\n")

    # get next page link
    nextPage = page_soup.findAll("a", {"aria-label": "Next"})[0]["href"]

    # get links to individual program download pages
    downloadPageHeaders = page_soup.findAll("h3", {"class": "search-head"})
    downloadPageLinks = []
    for pageHeader in downloadPageHeaders:
        pageHeader = pageHeader.findAll("a")[0]
        downloadPageLink = pageHeader["href"]
        print(downloadPageLink)
        downloadPageLinks.append(downloadPageLink)

    # load the page's linked download pages and download exe files
    for dlPage in downloadPageLinks:
        myUrl = dlPage
        myUrl = myUrl.replace("_program_", "-Download-Page-")

        # connecting to and downloading page
        uClient = uOpen(myUrl)
        page_html = uClient.read()
        uClient.close()

        # instatiating BeautifulSoup parsing of page
        dlPageSoup = soup(page_html, "html.parser")

        downLinks = dlPageSoup.findAll("a", {"class": "dwnlocations"})
        for link in downLinks:
            link = link["href"]
            try:
                file = uOpen(link)

                if int(file.info()['Content-Length']) <= 27000000:
                    print(str(count) + ": " + link)
                    os.system("sudo wget -O /media/lozmund/de9f2ab8-20e0-4d32-82a0-9564591262d0/home/freewareBenignFiles/" + str(count) + " " + link + " --read-timeout=1")
                    count += 1
            except:
                pass


    for pageNum in range(2,maxPage+1):
        print("Page " + str(pageNum) + ": ")
        myUrl = nextPage

        # connecting to and downloading page
        # last 8 characters of url is removed as it
        # didn't seem to effect loading of page, and
        # could not be parsed by 'urlopen' due to utf-8 encoding
        myUrl = myUrl[:-8]
        print("\n" + myUrl + "\n")
        uClient = uOpen(myUrl)
        page_html = uClient.read()
        uClient.close()

        # instatiating BeautifulSoup parsing of first page
        page_soup = soup(page_html, "html.parser")

        # get links to individual program download pages
        downloadPageHeaders = page_soup.findAll("h3", {"class": "search-head"})
        downloadPageLinks = []
        for pageHeader in downloadPageHeaders:
            pageHeader = pageHeader.findAll("a")[0]
            downloadPageLink = pageHeader["href"]
            print(downloadPageLink)
            downloadPageLinks.append(downloadPageLink)

        # load the page's linked download pages and download exe files
        for dlPage in downloadPageLinks:
            myUrl = dlPage
            myUrl = myUrl.replace("_program_","-Download-Page-")

            # connecting to and downloading page
            uClient = uOpen(myUrl)
            page_html = uClient.read()
            uClient.close()

            # instatiating BeautifulSoup parsing of page
            dlPageSoup = soup(page_html, "html.parser")

            downLinks = dlPageSoup.findAll("a", {"class": "dwnlocations"})
            for link in downLinks:
                link = link["href"]
                try:
                    file = uOpen(link)

                    if int(file.info()['Content-Length']) <= 27000000:
                        print(str(count) + ": " + link)
                        os.system(
                            "sudo wget -O /media/lozmund/de9f2ab8-20e0-4d32-82a0-9564591262d0/home/freewareBenignFiles/" + str(
                                count) + " " + link + " --read-timeout=1")
                        count += 1
                except:
                    pass

            # downLinks = dlPageSoup.findAll("a",{"class":"dwnlocations"})
            # for link in downLinks:
            #     link = link["href"]
            #     try:
            #         file = uOpen(link)
            #         print(file.info()['Content-Length'])
            #         if int(file.info()['Content-Length']) <= 27000000:
            #             print(str(count) + ": " + link)
            #             os.system("sudo wget -O ~/media/lozmund/de9f2ab8-20e0-4d32-82a0-9564591262d0/home/freewareBenignFiles/" + str(count) + " " + link)
            #             count += 1
            #     except:
            #         pass


        # get next page link
        nextPage = page_soup.findAll("a", {"aria-label": "Next"})[0]["href"]

