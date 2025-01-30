import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda") # for GPU


# Topics and keywords from our paper are below
bbc_news_topics = {
    'topic 0': ['said', 'mr', 'would', 'government', 'labour', 'us', 'also', 'year', 'election', 'new', 'blair',
                'people', 'minister', 'party', 'could'],
    'topic 1': ['club', 'chelsea', 'united', 'arsenal', 'liverpool', 'league', 'game', 'football', 'manager',
                'manchester', 'said', 'cup', 'players', 'would', 'goal'],
    'topic 2': ['film', 'best', 'actor', 'films', 'director', 'awards', 'award', 'actress', 'oscar', 'festival',
                'movie', 'aviator', 'said', 'year', 'star'],
    'topic 3': ['england', 'wales', 'ireland', 'rugby', 'france', 'nations', 'six', 'game', 'side', 'coach',
                'robinson', 'players', 'scotland', 'italy', 'first'],
    'topic 4': ['music', 'band', 'album', 'song', 'best', 'number', 'rock', 'singer', 'chart', 'one', 'said', 'us',
                'awards', 'top', 'pop'],
    'topic 5': ['open', 'roddick', 'seed', 'australian', 'match', 'set', 'win', 'nadal', 'tennis', 'first', '63',
                'beat', 'hewitt', 'final', 'federer'],
    'topic 6': ['broadband', 'phone', 'mobile', 'people', 'phones', 'bt', 'said', 'net', 'tv', 'services',
                'service', 'uk', 'digital', 'mobiles', 'million'],
    'topic 7': ['security', 'virus', 'software', 'users', 'email', 'spam', 'microsoft', 'said', 'site', 'windows',
                'net', 'attacks', 'viruses', 'data', 'spyware'],
    'topic 8': ['olympic', 'race', 'indoor', 'world', 'champion', 'holmes', 'championships', 'athens', 'european',
                'radcliffe', 'marathon', '60m', 'title', 'record', 'best'],
    'topic 9': ['games', 'game', 'gaming', 'nintendo', 'gamers', 'xbox', 'ds', 'video', 'sony', 'titles', 'console',
                'play', 'said', 'halo', 'time'],
    'topic 10': ['show', 'tv', 'series', 'bbc', 'said', 'channel', 'comedy', 'television', 'audience', 'viewers',
                 'celebrity', 'us', 'also', 'programme', 'million'],
    'topic 11': ['music', 'files', 'digital', 'said', 'technology', 'p2p', 'players', 'bittorrent', 'filesharing',
                 'apple', 'networks', 'software', 'people', 'piracy', 'peertopeer'],
    'topic 12': ['gadgets', 'gadget', 'digital', 'technology', 'technologies', 'show', 'devices', 'people',
                 'electronics', 'sony', 'mobile', 'consumer', 'video', 'make', 'laptop'],
    'topic 13': ['search', 'google', 'blogs', 'web', 'yahoo', 'blog', 'people', 'users', 'microsoft', 'jeeves',
                 'information', 'ask', 'desktop', 'said', 'internet'],
    'topic 14': ['kenteris', 'iaaf', 'thanou', 'greek', 'drugs', 'conte', 'athens', 'doping', 'test', 'tests',
                 'olympics', 'balco', 'athletes', 'athletics', 'banned']
}

arxiv_abstracts_topics = {
    'topic 0': ['mass', 'ray', 'energy', 'model', 'observations', 'data', '10', 'star', 'emission', 'high',
                'observed', 'galaxies', 'stars', 'dark', 'solar'],
    'topic 1': ['learning', 'data', 'training', 'based', 'model', 'neural', 'image', 'models', 'methods',
                'performance', 'network', 'method', 'propose', 'images', 'deep'],
    'topic 2': ['spin', 'quantum', 'magnetic', 'graphene', 'phase', 'field', 'temperature', 'states', 'state',
                'two', 'energy', 'electron', 'transition', 'density', 'model'],
    'topic 3': ['group', 'graph', 'prove', 'algebra', 'groups', 'graphs', 'show', 'algebras', 'finite', 'number',
                'let', 'paper', 'every', 'give', 'set'],
    'topic 4': ['theory', 'black', 'gauge', 'quark', 'hole', 'field', 'pi', 'string', 'brane', 'decays', 'qcd',
                'scalar', 'theories', 'mass', 'two'],
    'topic 5': ['random', 'stochastic', 'time', 'model', 'networks', 'process', 'network', 'distribution',
                'processes', 'probability', 'dynamics', 'show', 'results', 'study', 'two'],
    'topic 6': ['solutions', 'equation', 'equations', 'prove', 'space', 'spaces', 'existence', 'paper', 'solution',
                'boundary', 'infinity', 'problem', 'operator', 'type', 'case'],
    'topic 7': ['learning', 'algorithm', 'policy', 'control', 'problem', 'algorithms', 'reinforcement', 'agent',
                'rl', 'optimal', 'optimization', 'agents', 'robot', 'based', 'reward'],
    'topic 8': ['method', 'numerical', 'problems', 'convergence', 'problem', 'methods', 'order', 'algorithm',
                'convex', 'element', 'optimization', 'finite', 'linear', 'solution', 'error'],
    'topic 9': ['quantum', 'classical', 'states', 'qubit', 'state', 'qubits', 'entanglement', 'algorithm',
                'circuit', 'error', 'protocol', 'key', 'circuits', 'gates', 'qkd'],
    'topic 10': ['liquid', 'dynamics', 'model', 'phase', 'simulations', 'surface', 'stress', 'transition',
                 'granular', 'shear', 'two', 'energy', 'temperature', 'particles', 'fluid'],
    'topic 11': ['channel', 'codes', 'mimo', 'interference', 'multiple', 'proposed', 'performance', 'rate', 'power',
                 'user', 'wireless', 'communication', 'paper', 'users', 'capacity'],
    'topic 12': ['privacy', 'data', 'federated', 'blockchain', 'fl', 'performance', 'learning', 'network',
                 'security', 'based', 'paper', 'clients', 'model', 'private', 'attacks'],
    'topic 13': ['optical', 'photon', 'quantum', 'mode', 'light', 'frequency', 'photonic', 'states', 'modes',
                 'nonlinear', 'two', 'phase', 'single', 'wave', 'state'],
    'topic 14': ['regression', 'data', 'estimator', 'estimators', 'treatment', 'causal', 'estimation',
                 'distribution', 'model', 'bayesian', 'methods', 'inference', 'method', 'proposed', 'sample'],
    'topic 15': ['plasma', 'turbulence', 'flow', 'turbulent', 'magnetic', 'velocity', 'field', 'reynolds',
                 'simulations', 'energy', 'scale', 'fluid', 'plasmas', 'numerical', 'flows'],
    'topic 16': ['software', 'code', 'students', 'research', 'ai', 'development', 'learning', 'based', 'data',
                 'citation', 'paper', 'developers', 'process', 'science', 'study'],
    'topic 17': ['clustering', 'data', 'algorithm', 'tree', 'phylogenetic', 'trees', 'hashing', 'algorithms',
                 'time', 'problem', 'based', 'persistence', 'methods', 'distance', 'space'],
    'topic 18': ['power', 'grid', 'system', 'energy', 'voltage', 'battery', 'renewable', 'proposed', 'control',
                 'electricity', 'load', 'charging', 'demand', 'storage', 'model'],
    'topic 19': ['traffic', 'vehicles', 'vehicle', 'driving', 'lane', 'model', 'autonomous', 'prediction', 'road',
                 'time', 'based', 'pedestrian', 'flow', 'network', 'trajectory'],
    'topic 20': ['imaging', 'resolution', 'microscopy', 'image', 'phase', 'images', 'optical', 'ultrasound',
                 'diffraction', 'light', 'speckle', 'method', 'reconstruction', 'high', 'scattering'],
    'topic 21': ['withdrawn', 'paper', 'author', 'comment', 'authors', 'phys', 'due', 'university', 'reply',
                 'arxiv', 'rev', 'professor', 'lett', 'error', 'cond'],
    'topic 22': ['fractional', 'derivative', 'order', 'caputo', 'derivatives', 'numerical', 'equations',
                 'differential', 'method', 'time', 'equation', 'scheme', 'solution', 'alpha', 'calculus'],
    'topic 23': ['players', 'team', 'player', 'teams', 'sports', 'football', 'game', 'league', 'soccer', 'season',
                 'games', 'performance', 'basketball', 'data', 'matches'],
    'topic 24': ['localization', 'indoor', 'positioning', 'accuracy', 'based', 'location', 'fingerprinting', 'wifi',
                 'rss', 'fingerprint', 'signal', 'uwb', 'wireless', 'rssi', 'proposed']
}

amazon_reviews_topics = {
    'topic -1': ['great', 'sound', 'good', 'use', 'one', 'quality', 'works', 'like', 'would', 'product', 'it',
                 'well', 'work', 'camera', 'get'],
    'topic 0': ['keyboard', 'mouse', 'router', 'one', 'monitor', 'wifi', 'cable', 'use', 'great', 'works', 'work',
                'keys', 'it', 'would', 'like'],
    'topic 1': ['sound', 'radio', 'great', 'quality', 'good', 'antenna', 'one', 'music', 'headphones', 'alexa',
                'use', 'get', 'like', 'would', 'echo'],
    'topic 2': ['case', 'tablet', 'kindle', 'ipad', 'screen', 'cover', 'it', 'like', 'love', 'one', 'great', 'fire',
                'would', 'fits', 'use'],
    'topic 3': ['battery', 'charge', 'charger', 'watch', 'batteries', 'charging', 'cord', 'phone', 'one', 'great',
                'gps', 'cords', 'use', 'garmin', 'works'],
    'topic 4': ['camera', 'lens', 'cameras', 'tripod', 'great', 'it', 'use', 'one', 'good', 'quality', 'get',
                'would', 'well', 'like', 'video'],
    'topic 5': ['bag', 'great', 'backpack', 'price', 'good', 'love', 'quality', 'color', 'laptop', 'well',
                'product', 'nice', 'sturdy', 'cover', 'works'],
    'topic 6': ['works', 'great', 'love', 'product', 'expected', 'perfect', 'advertised', 'good', 'it', 'needed',
                'thanks', 'exactly', 'worked', 'perfectly', 'thank'],
    'topic 7': ['drive', 'fan', 'card', 'laptop', 'fans', 'computer', 'ssd', 'one', 'drives', 'case', 'memory',
                'power', 'ram', 'great', 'it'],
    'topic 8': ['product', 'fast', 'working', 'great', 'shipping', 'months', 'works', 'stopped', 'arrived', 'item',
                'worked', 'delivery', 'one', 'service', 'work'],
    'topic 9': ['tv', 'remote', 'projector', 'picture', 'hdmi', 'one', 'fire', 'great', 'roku', 'cable', 'stick',
                'screen', 'use', 'amazon', 'quality'],
    'topic 10': ['easy', 'install', 'set', 'use', 'great', 'works', 'setup', 'product', 'instructions', 'up',
                 'assemble', 'good', 'installation', 'it', 'installed'],
    'topic 11': ['mount', 'tv', 'stand', 'wall', 'monitor', 'desk', 'screws', 'sturdy', 'easy', 'monitors',
                 'mounting', 'one', 'great', 'would', 'bracket'],
    'topic 12': ['band', 'fitbit', 'bands', 'strap', 'wrist', 'watch', 'comfortable', 'love', 'ties', 'like', 'one',
                 'fit', 'wear', 'original', 'great'],
    'topic 13': ['stickers', 'air', 'label', 'bubbles', 'clean', 'dust', 'product', 'sticker', 'great', 'labels',
                 'plastic', 'cleaning', 'stick', 'adhesive', 'use'],
    'topic 14': ['light', 'lights', 'bulb', 'bright', 'lamp', 'lighting', 'led', 'use', 'one', 'great',
                 'brightness', 'bulbs', 'easy', 'ring', 'like'],
    'topic 15': ['fit', 'fits', 'perfect', 'perfectly', 'great', 'well', 'looks', 'good', 'works', 'install',
                 'head', 'product', 'easy', 'harness', 'nice'],
    'topic 16': ['water', 'waterproof', 'shower', 'camera', 'underwater', 'pool', 'speaker', 'great', 'swimming',
                 'use', 'sound', 'fish', 'pictures', 'snorkeling', 'good'],
    'topic 17': ['printer', 'print', 'ink', 'printing', 'hp', 'cartridges', 'scanner', 'cartridge', 'printers',
                 'prints', 'paper', 'printed', 'work', 'one', 'epson'],
    'topic 18': ['protection', 'lock', 'security', 'protective', 'protects', 'protect', 'good', 'great', 'locks',
                 'product', 'system', 'well', 'home', 'it', 'price'],
    'topic 19': ['dislike', 'nothing', 'dislikes', 'disliked', 'product', 'anything', 'hate', 'liked', 'it', 'like',
                 'love', 'everything', 'dont', 'fact', 'works']
}

newsgroup20_topics = {
    'topic 0': ['game', 'team', 'year', 'games', 'hockey', 'players', 'season', 'play', 'writes', 'baseball',
                'last', 'league', 'win', 'player', 'would'],
    'topic 1': ['israel', 'people', 'one', 'writes', 'would', 'article', 'god', 'israeli', 'jews', 'think', 'say',
                'arab', 'believe', 'it', 'right'],
    'topic 2': ['drive', 'scsi', 'card', 'windows', 'disk', 'modem', 'ide', 'drives', 'controller', 'use',
                'problem', 'dos', 'hard', 'system', 'bus'],
    'topic 3': ['gun', 'people', 'would', 'fbi', 'guns', 'think', 'fire', 'president', 'writes', 'article', 'batf',
                'one', 'koresh', 'government', 'right'],
    'topic 4': ['windows', 'window', 'jpeg', 'file', 'image', 'use', 'files', 'color', 'graphics', 'program',
                'display', 'gif', 'version', 'os', 'format'],
    'topic 5': ['space', 'launch', 'nasa', 'orbit', 'earth', 'would', 'shuttle', 'moon', 'solar', 'mission',
                'spacecraft', 'satellite', 'writes', 'lunar', 'like'],
    'topic 6': ['key', 'encryption', 'clipper', 'chip', 'keys', 'government', 'privacy', 'security', 'escrow',
                'use', 'des', 'nsa', 'would', 'secure', 'algorithm'],
    'topic 7': ['sale', '00', 'offer', 'price', 'shipping', 'please', 'asking', 'condition', 'drive', 'mail', 'new',
                'cd', 'sell', '10', 'interested'],
    'topic 8': ['bike', 'dod', 'writes', 'dog', 'ride', 'article', 'riding', 'motorcycle', 'like', 'helmet', 'get',
                'bikes', 'one', 'car', 'front'],
    'topic 9': ['msg', 'patients', 'medical', 'disease', 'doctor', 'food', 'cancer', 'treatment', 'one', 'candida',
                'pain', 'yeast', 'health', '92', 'vitamin'],
    'topic 10': ['god', 'jesus', 'bible', 'christ', 'church', 'one', 'would', 'mary', 'christian', 'heaven', 'hell',
                 'people', 'faith', 'sin', 'believe'],
    'topic 11': ['car', 'cars', 'engine', 'ford', 'dealer', 'writes', 'price', 'article', 'new', 'mustang', 'oil',
                 'would', 'like', 'miles', 'one'],
    'topic 12': ['mail', 'address', 'comp', 'list', 'gopher', 'ftp', 'software', 'please', 'email', 'system',
                 'group', 'space', 'files', 'graphics', 'mac'],
    'topic 13': ['printer', 'print', 'fonts', 'hp', 'font', 'deskjet', 'printers', 'laser', 'ink', 'windows',
                 'printing', 'postscript', 'use', 'paper', 'truetype'],
    'topic 14': ['armenian', 'turkish', 'armenians', 'armenia', 'turks', 'turkey', 'people', 'said', 'soviet',
                 'azerbaijan', 'genocide', 'greek', 'russian', 'one', 'greece'],
    'topic 15': ['radar', 'detector', 'detectors', 'car', 'receiver', 'alarm', 'radio', 'would', 'use', 'one',
                 'get', 'transmitter', 'antenna', 'valentine', 'writes'],
    'topic 16': ['polygon', 'points', 'algorithm', 'polygons', 'sphere', 'line', 'problem', 'lines', 'point',
                 'plane', 'surface', 'routine', 'p1', 'convex', 'xxxx'],
    'topic 17': ['insurance', 'health', 'private', 'care', 'canada', 'geico', 'canadian', 'system', 'coverage',
                 'doctors', 'hospital', 'medical', 'pay', 'americans', 'writes'],
    'topic 18': ['monitor', 'tempest', 'power', 'monitors', 'computer', 'computers', 'hours', 'turn', 'electricity',
                 'day', '24', 'equipment', 'pick', 'emissions', 'consumption'],
    'topic 19': ['battery', 'concrete', 'batteries', 'discharge', 'acid', 'lead', 'temperature', 'floor', 'dirt',
                 'reaction', 'heat', 'stored', 'electrolyte', 'terminals', 'garage'],
    'topic 20': ['cpu', 'fan', 'heat', 'sink', 'fans', 'power', 'chip', 'idle', 'cooling', 'hot', 'computationally',
                 'running', 'intensive', 'case', 'supply'],
    'topic 21': ['photography', 'kirlian', 'pictures', 'krillean', 'leaf', 'aura', 'object', 'corona', 'spelling',
                 'energy', 'involves', 'taking', 'sci', 'plates', 'huey'],
    'topic 22': ['blue', 'uv', 'boards', 'leds', 'light', 'led', 'green', 'solder', 'mask', 'bulb', 'emit', 'board',
                 'bulbs', 'colour', 'seen'],
    'topic 23': ['cancer', 'cholesterol', 'medical', 'circumcision', 'diet', 'health', 'pregnancy', 'teacher',
                 'drug', 'fat', 'disease', 'biology', 'sperm', 'risk', 'birth'],
    'topic 24': ['paint', 'wax', 'finish', 'car', 'scratches', 'rowlands', 'lisa', 'buff', 'dull', 'plastic',
                 'good', 'scuffed', 'fiance', 'bike', 'black']
}

worldcup_tweets_topics = {
    'topic 0': ['cup', 'world', 'worldcup', 'rt', 'team', '2022', 'fifa', 'qatar', 'live', 'win', 'nft', 'football',
                'worldcup2022', 'join', 'championship'],
    'topic 1': ['amp', 'follow', 'hours', 'proof', 'rt', 'join', '24', 'international', 'icc', '100', '40rt', '80',
                '40', '24350', '477'],
    'topic 2': ['optout', 'reply', 'thanks', 'unsubscribe', 'fifaworldcuponfox', 'tweet', 'crazyfootballongotv',
                'usa', 'playing', 'optingin', 'well', 'biggest', 'stop', 'enjoy', 'tournament'],
    'topic 3': ['airdrop', '000', 'token', '10', 'airdrops', 'join', '120', 'lfg', 'fb', 'biggest', 'msp',
                'metasports', 'msg', 'us', 'progress'],
    'topic 4': ['katara', 'hayya', 'karwa', 'fiverr', 'aang', 'zuko', 'editing', 'khor', 'photo', 'arhbo',
                'studios', 'card', 'directed', 'fi', 'light'],
    'topic 5': ['prediction', 'share', 'nftplayer', 'constantly', 'project', 'para', 'vote', 'voting', 'update',
                'latest', 'world', 'score', 'make', 'nft', 'football'],
    'topic 6': ['mls', 'series', 'phillies', 'philadelphia', 'union', 'lose', 'bowl', 'philly', 'super', 'day',
                'tough', 'eagles', 'lost', 'fans', 'superbowl'],
    'topic 7': ['x7', 'x8', 'x4', 'x3', 'league', 'rey', 'champions', 'del', 'pr', 'club', 'liga', 'copa',
                'european', 'la', 'sure'],
    'topic 8': ['concentration', 'camps', 'poster', '1978', 'arg', 'calling', 'protest', 'french', 'boycott', 'no',
                'football', 'rt', 'cup', 'world', 'coba'],
    'topic 9': ['lecturing', 'urging', 'morality', 'written', 'focus', 'part', 'teams', 'tournament', 'fifa', 'rt',
                'cup', 'world', 'dr', 'moralsk', 'overg'],
    'topic 10': ['dapp', 'gttooneychain', 'immersion', 'tooneychain', 'introduction', 'environment', 'thread',
                 'better', 'rt', 'nws', 'vry', 'thy', 'signal', 'pump', 'announced'],
    'topic 11': ['gunathilaka', 'danushka', 'sri', 'assault', 'sexual', 'cricketer', 'sydney', 'arrested', 'lankan',
                 'lanka', 'charged', 'rape', 'alleged', 't20', 'danushkagunathilaka'],
    'topic 12': ['fiverr', 'editing', 'photo', 'careem', 'webdesign', 'graphicdesign', 'removal', 'satisfaction',
                 'photography', 'guarantee', 'background', 'sharp', 'contact', 'service', 'amazing'],
    'topic 13': ['virat', 'ponting', 'kohli', 'said', 'ricky', 'rauf', 'talked', 'remembered', 'trem', 'haris',
                 'six', 'looked', 'brings', 'knew', 'asia'],
    'topic 14': ['vinyl', 'vinylcollection', 'vinylcommunity', 'groovevinylstore', 'vintage', 'records', 'music',
                 'groovevinyl', 'imacelebrity', 'nowspinning', 'imaceleb', 'blackfriday', 'via', 'vinylrecords',
                 'thursdayvibes']
}

# dataset descriptions are below for each model
bbc_news_description = 'The dataset used comprised of 2,225 bbc news articles from the time of 2004-2005. The articles cover across five news categories: business, entertainment, politics, sport, and tech.'

arxiv_description = 'This dataset comprises 2,521,247 abstracts of scientific articles from the arXiv repository, summarizing research across disciplines such as physics, computer science, mathematics, statistics, electrical engineering, quantitative biology, and economics. The content is highly technical and research-focused, providing concise summaries of complex studies.'

amazon_reviews_description = 'This dataset contains user-generated reviews from the electronics category on Amazon, featuring opinions, ratings, and experiences related to various electronic products. The reviews reflect consumer feedback on product quality, performance, and usability.'

news_description = 'The 20 Newsgroups dataset comprises 18,846 documents from various Usenet newsgroups collected between 1993 and 1994. These documents span 20 different topics, ranging from technical discussions to social commentary and debates on a wide array of subjects.'

worldcup_description = 'This dataset consists of 2,407,396 tweets collected using the WorldCup2022 hashtag on Twitter, spanning from November 1, 2022, to January 9, 2023.'

topic_dicts = [("bbc_news", bbc_news_topics, bbc_news_description),
               ("arxiv_abstracts", arxiv_abstracts_topics, arxiv_description),
               ("amazon_reviews", amazon_reviews_topics, amazon_reviews_description),
               ("newsgroup20", newsgroup20_topics, news_description),
               ("worldcup_tweets", worldcup_tweets_topics, worldcup_description)]

def llm_labeling(user_message, model_name, model=None, tokenizer=None, system_message=None, openai_client=None):
    """
    This function implements the labeling workflow described in: "Empowering Topic Modeling with Large Language Models (LLMs): A Comparative Study on Labeling Efficiency and Accuracy."

    params:
        user_message: the message that describes the message to the LLM
        model_name: the name of the model (used for our purposes and the 3 models used in the paper)
        model: the LLM model object. If using GPT, no need to pass
        tokenizer: the tokenizer used to tokenize the message
        system_message: the message that describes the message to the system - can leave empty if model does not take one
        openai_client: client to use for openai models
    """
    llm_generated_labels = {}

    for dataset_name, topic, dataset_description in topic_dicts: # tuple of dataset name, dataset topics and keywords, and then the dataset description
        llm_generated_labels[dataset_name] = {}
        for topic_num, topic_keywords in topic.items():

            # format the messages with the system and user message
            formatted_system_message = system_message.format(dataset_description=dataset_description)
            formatted_user_message = user_message.format(topic_keywords=', '.join(topic_keywords))

            if model_name == 'llama_3_1_8b_instruct' or model_name == 'llama_3_2_3b_instruct':
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    {formatted_system_message}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    
    {formatted_user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>""" # prompt with llama formatting

                input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

                output = model.generate(**input_ids, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id,
                                        temperature=0.5)
                text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

                # extract just the assistant's response, excluding the prompt
                assistant_index = text.find("assistant")
                if assistant_index != -1:
                    text = text[assistant_index + len("Assistant:"):].strip()
                else:
                    text = "No valid label generated"

                llm_generated_labels[dataset_name][topic_num] = text # adding label to results dictionary

            elif model_name == 'gpt-4o':

                output = openai_client.chat.completions.create( # using openai api
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": formatted_system_message},
                        {"role": "user", "content": formatted_user_message}
                    ]).choices[0].message.content

                llm_generated_labels[dataset_name][topic_num] = output # adding label to results dictionary

    df = pd.DataFrame([ # formatting dataframe to save to csv
        {"Dataset": dataset, "Topic": topic, "Label": label}
        for dataset, topics in llm_generated_labels.items()
        for topic, label in topics.items()
    ])

    df.to_csv(f'{model_name}_labeling_results.csv', index=False)

    print("success for model")

def test():
    """
    This is a tester function using the LLM labeling workflow function and using the same models, data, and system/user messages as the paper
    """
    system_message = "You are an expert topic modeler. You are tasked with generating a topic label that is both clear and contextually relevant, accurately reflecting the theme of a topic based on its keywords. {dataset_description} Your label should succinctly capture the essence of the topic, considering the broader context of the dataset while remaining focused and precise. The label should be clear, concise (not too long), and descriptive, encapsulating the core subject of the topic in a way that is easily understood. "
    user_message = "The keywords for this topic are: {topic_keywords}. Provide a label that best captures the theme of the topic suggested by these keywords. Provide one label only with no additional text."

    llama3_1_8b_token = '' # need to provide own llama token from hugging face for llama 3.1 8b instruct
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=llama3_1_8b_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=llama3_1_8b_token).to('cuda')
    llm_labeling(user_message, 'llama_3_1_8b_instruct', model, tokenizer, system_message)
    del model, tokenizer
    torch.cuda.empty_cache()

    llama3_2_3b_token = '' # need to provide own llama token from hugging face for llama 3.3 3b instruct
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=llama3_1_8b_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=llama3_2_3b_token).to('cuda')
    llm_labeling(user_message, 'llama_3_2_3b_instruct', model, tokenizer, system_message)
    del model, tokenizer
    torch.cuda.empty_cache()

    client = OpenAI(api_key='') # need to provide own api key for openai
    llm_labeling(user_message, 'gpt-4o', system_message=system_message, openai_client=client)
    del client
    torch.cuda.empty_cache()

if __name__ == '__main__':
    test()
