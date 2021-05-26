# from pubtatortool import PubTatorCorpus
import json 

class MedMentionsDB(object):
    # our dataset will look like 
    # {
    # PMID: {
    #     "title":"",
    #     "abstract":"",
    #     "entities":[
    #           (StartIndex, EndIndex, MentionTextSegment, SemanticTypeID, EntityID),
    #           (StartIndex, EndIndex, MentionTextSegment, SemanticTypeID, EntityID)
    #      ]
    #   }
    # ,...
    # }
    def __init__(self, data_file_path):
        self._data_file_path = data_file_path

        self.language = "eng"
        self.pmids = []
        self.data = {}

        self.entity_to_pmid = {}

        # read in a previously created DB file
        with open(data_file_path) as f:
            self.data = json.load(f)

        # we will need a reverse dictionary to go from the entity IDs to PMIDs that they came from
        for pmid, elem in self.data.items():
            for entity in elem['entities']:
                self.entity_to_pmid[entity[4]] = pmid

    def get_pmids(self):
        if self.pmids:
            return self.pmids

        else:
            with open(self._data_file_path) as f:
                ids = json.load(f)
                self.pmids = list(ids)
                return self.pmids

    def page_size(self):
        '''
        This function returns the number of papers in the database
        '''
        if self.data:
            return len(self.data.keys())

        else:
            with open(self._data_file_path) as f:
                ids = json.load(f).keys()
                self.pmids = ids
                return len(ids)

    def get_data(self):
        return self.data

    @staticmethod
    def build(input_data_file, out_file, 
           # pool_size, chunk_size, preprocess_func=None,
           # init_map_size=500000000, buffer_size=3000
            ):

        # corpus = PubTatorCorpus(self._data_file_name)
        data = {}
        
        with open(input_data_file) as f:
            fdata = f.read()
            example_list = fdata.split("\n\n")
        

        for article in example_list:
            article_data = article.split("\n")
            # the first element is the title
            # second is abstract
            # the rest are mentions
            title = article_data[0].split("|") 
            abstract = article_data[1].split("|")
            entities = []
            for entity in article_data[2:]:
                entity_entry = entity.split("\t")[1:]
                entity_entry[0] = int(entity_entry[0])
                entity_entry[1] = int(entity_entry[1])
                entities.append(entity_entry)
            
            pmid = title[0]
            data[pmid] = {
                "title": title[2],
                "abstract": abstract[2],
                "entities": entities
            }

        
        with open(out_file, 'w') as outfile:
            json.dump(data, outfile)
        

        
