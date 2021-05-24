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
    def __init__(self, data_file_path, id_file_path):
        self._data_file_path = data_file_path
        self._id_file_path = id_file_path        

        self.language = "eng"
        self.pmids = []
        self.data = {}

    def pmids(self):
        if self.titles:
            return self.titles

        else:
            with open(self._id_file_path) as f:
                ids = f.read().splitlines()
                self.pmids = ids
                return ids

    def page_size(self):
        pass


    @staticmethod
    def build(input_data_file, out_file, 
           # pool_size, chunk_size, preprocess_func=None,
           # init_map_size=500000000, buffer_size=3000
            ):

        # corpus = PubTatorCorpus(self._data_file_name)
        data = {}
        
        with open() as f:
            fdata = f.read(input_data_file)
            example_list = fdata.split("\n\n")
        

        for article in example_list:
            article_data = article.split("\n")
            # the first element is the title
            # second is abstract
            # the rest are mentions
            title = article_data[0].split("|") 
            abstract = article_data[1].split("|")
            entities = [tuple(entity.split("|")) for entity in article_data[2:]]
            pmid = title[0]
            data[int(pmid)] = {
                "title": title[2],
                "abstract": abstract[2],
                "entities": entities
            }

        
        with open(out_file, 'w') as outfile:
            json.dump(data, outfile)
        

        
