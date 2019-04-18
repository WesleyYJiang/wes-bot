import requests
from bs4 import BeautifulSoup

base_url = "http://officequotes.net"

season, episode = 1, 1
episode_url = f"{base_url}/no{season}-{episode:02}.php"
# result = requests.get(episode_url)
# soup = BeautifulSoup(result.content, "lxml")
#
# quotes = soup.find_all("div", class_="quote")
# print(len(quotes))
# soup.get_text()


# Try selenium, doesn't look like I am getting the full data from the above method
from selenium import webdriver

# make sure installation is completed correctly
# https://selenium-python.readthedocs.io/installation.html
# driver = webdriver.Chrome()
# driver.implicitly_wait(5)
# driver.get("https://www.springfieldspringfield.co.uk/view_episode_scripts.php?tv-show=the-office-us&episode=s01e01")
# quotes = driver.find_elements_by_class_name("scrolling-script-container")
# driver.implicitly_wait(5)
# print(len(quotes))
#
#
# print(quotes[-1].text)



def write_episode_to_file(season, episode):
    base_url = "https://www.springfieldspringfield.co.uk/view_episode_scripts.php?tv-show=the-office-us&episode="
    episode_url = f"{base_url}s{season:02}e{episode:02}"
    print(episode_url)
    driver = webdriver.Chrome()
    driver.implicitly_wait(2)
    driver.get(episode_url)
    driver.implicitly_wait(2)
    quotes = driver.find_elements_by_class_name("scrolling-script-container")

    with open("the-office-all-episodes.txt", "at") as f:
        for quote in quotes:
            f.write(quote.text + '\n\n')
    driver.close()

num_episodes_per_season = [8, 22, 23, 14, 26, 24, 24, 24, 23]

for season, num_episodes in enumerate(num_episodes_per_season):
    for episode in range(num_episodes):
        write_episode_to_file(season+1, episode+1)
