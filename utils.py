from tqdm.auto import tqdm
from duckduckgo_search import DDGS

def collect_images(keywords: str, 
                   path: str,
                   max_results: int = 30,
                   timeout: tuple = (3, 5)):
  Path(path).mkdir(parents=True, exist_ok=True)
  image_results = []
  with DDGS() as ddgs:
    results = list(ddgs.images(keywords=keywords, max_results=max_results))
  for i, result in enumerate(tqdm(results)):
    try:
      r = requests.get(result['image'], timeout=timeout)
    except:
      continue
    if r.status_code == 200: 
      with open(f'{path}/{i}.jpg','wb') as img: 
        img.write(r.content)
      image_result = {'image': result['image'],
                      'url': result['url'],
                      'path': f'{path}/{i}.jpg',
                      'height': result['height'],
                      'width': result['width'],
                      'source': result['source']}
      image_results.append(image_result)
  print(f'[INFO] Downloaded {len(image_results)} images into {path}')
  return image_results