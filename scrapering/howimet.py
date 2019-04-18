from selenium import webdriver

def write_episode_to_file(season, episode):
    base_url = "https://www.springfieldspringfield.co.uk/view_episode_scripts.php?tv-show=how-i-met-your-mother&episode="
    episode_url = f"{base_url}s{season:02}e{episode:02}"
    print(episode_url)
    driver = webdriver.Chrome()
    driver.implicitly_wait(2)
    driver.get(episode_url)
    driver.implicitly_wait(2)
    quotes = driver.find_elements_by_class_name("scrolling-script-container")

    with open("howimetyourmother.txt", "at") as f:
        for quote in quotes:
            f.write(quote.text + '\n\n')
    driver.close()

num_episodes_per_season = [22, 22, 20, 24, 24, 24, 24, 24, 24]

for season, num_episodes in enumerate(num_episodes_per_season):
    for episode in range(num_episodes):
        write_episode_to_file(season+1, episode+1)
