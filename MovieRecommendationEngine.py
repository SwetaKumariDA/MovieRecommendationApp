import pandas as pd
import json
import nltk as npl
from sklearn import preprocessing

class Dataload():
    filepath_moviesData='tmdb_5000_movies.csv'
    filepath_creditsData='tmdb_5000_credits.csv'
    json_col_list=['genres', 'keywords', 'production_countries','production_companies', 'spoken_languages','cast', 'crew']
    
    
    def __load_IMDBdata(self,filepath):
        """The function loads IMDB movie data into pandas data tables
        """
        try:
            self.IMDBdata = pd.read_csv(filepath)
            return self.IMDBdata
        except:
            print("File Not found")
    
    def __rename_col(self,df):
        """The function renames 'Id' column of moviesData to movie_id.it ir required for 
        merging tmdb_5000_movies.csv and tmdb_5000_credits.csv data on the basis of movie id key
        """ 
        df.rename(columns={'id':'movie_id'}, inplace=True)
        
      
    def data_merge(self):
        """The function merges movies and credit data
        """
        self.moviesData=self.__load_IMDBdata(Dataload.filepath_moviesData)
        self.creditsData=self.__load_IMDBdata(Dataload.filepath_creditsData)
        self.__rename_col(self.moviesData)
        self.merge_Final_data=pd.merge(self.moviesData,self.creditsData,on='movie_id')
        return self.merge_Final_data
    
    def __getting_list(self,json_col):
        """The method extacts desired data from a json column.
        It extracts the data then get the set of unique values and finally joins the data.
        """
        try:
            rt=",".join(map(str,list(set([x['name'].strip() for x in json_col]))))
        except IndexError or KeyError:
            rt=pd.np.nan
            
        return rt

    
    def __movie_cast(self,cast):
        """The method extacts desired data from  json column 'cast'.
        It extracts the top three actors name and finally joins the data.
        """ 
        try:
            top_movie_cast =",".join(map(str,list(set([x['name'] for x in cast if x['order'] in (0,1,2)]))))
        except IndexError or KeyError:
            top_movie_cast=pd.np.nan
        return top_movie_cast

    def __movie_directors(self,crew_data):
        """The method extacts desired data from  json column 'crew_data'.
        It extracts name of all the directors.
        """ 
        try:
            dir=",".join(map(str,list(set([x['name'] for x in crew_data if x['job'] == 'Director']))))
        except IndexError or KeyError:
            dir=pd.np.nan
        
        return dir


    def __movie_name(self,original_title):
        x=original_title
        x=x.strip()
        x=x.lower()
        return x 


    def format_Data(self):
        """The function fetches the information from json columns
        """
        moviesData=self.data_merge()
        moviesData['release_date'] = pd.to_datetime(moviesData['release_date']).apply(lambda x: x.date())
        for d in Dataload.json_col_list:
            moviesData[d] =moviesData[d].apply(json.loads)
        moviesData['genres'] = moviesData['genres'].apply(self.__getting_list)
        moviesData['keywords'] = moviesData['keywords'].apply(self.__getting_list)
        moviesData['production_countries'] = moviesData['production_countries'].apply(self.__getting_list)
        moviesData['production_companies'] = moviesData['production_companies'].apply(self.__getting_list)
        moviesData['spoken_languages'] = moviesData['spoken_languages'].apply(self.__getting_list)
        moviesData['movie_directors'] = moviesData['crew'].apply(self.__movie_directors)
        moviesData['top_movie_cast'] = moviesData['cast'].apply(self.__movie_cast)
        moviesData['original_title'] = moviesData['original_title'].apply(self.__movie_name)
        return moviesData
 



class NlpWordStemming():
    def __init__(self):
        self.movie_stem = npl.stem.PorterStemmer()
    def __list1(self,words):
        words=words.split(',')
        return ",".join(map(str,list(set([self.movie_stem.stem(word) for word in words]))))

    def __map_similar_words(self,format_Data):
        formatted_data=format_Data
        formatted_data['new_keywords']=formatted_data['keywords'].apply(self.__list1)
        return formatted_data

    def movie_data(self,format_Data):
        dtm=self.__map_similar_words(format_Data)
        return dtm


#-------------------------------Checking Missing values-------------------------------
#obj1= Dataload()
#x2=obj1.format_Data()
#percent =(x2.isnull().sum()/x2.shape[0])*100
#print(percent .sort_values(ascending=False))
'''
As percentage of missing value is very low and missing values are present 
in columns which are not considered in our movie recommendation, no need to do anything
'''

#-------------------------------Checking the keywords data for similar meaning words-------------------------------
'''
keyword_dict is a dictionary which gives count of every word present in the keyword colums
In keyword data we can see there are many words which have the similar meaning.
Hence need to have stem word for those similar words.
'''

'''
keyword_dict={}
def word_mapping(dt):
    for item in dt:
        words=item.split(',')
        for word in words:
            word=word.strip()
            keyword_dict[word]=keyword_dict.get(word,0)+1
    keyword_table=pd.DataFrame(keyword_dict,index=['Count'])
    keyword_table=keyword_table.T
    keyword_table=keyword_table.sort_values(by=['Count'],ascending=False)
    
    return keyword_table
'''
#df=word_mapping(x2['keywords'])
#df.head(15)


#obj2=NlpWordStemming()
#x3=obj2.movie_data(x2)

'''Here we can see count for few words have increased, it means our stemmer is working. Example: count for 'murder ' and 
musical words before stemming was 189 and 105 .after using nlp count changes to 196 and 139 respectively
'''
'''
keyword_dict={}
def word_mapping(dt):
    for item in dt:
        words=item.split(',')
        for word in words:
            word=word.strip()
            keyword_dict[word]=keyword_dict.get(word,0)+1
    keyword_table=pd.DataFrame(keyword_dict,index=['Count'])
    keyword_table=keyword_table.T
    keyword_table=keyword_table.sort_values(by=['Count'],ascending=False)
    
    return keyword_table
'''
#word_map=word_mapping(x3['new_keywords'])
#word_map.head(10)

#------------------------------- End -------------------------------


class MovieScore():
    
    def __init__(self,merge_Final_data,user_inputs,userstr):
       self.merge_Final_data=merge_Final_data
       self.user_inputs=user_inputs
       self.userstr=userstr
        
    
    def __genres_score(self,coldata):
        '''
        The method counts the number of genres matched with genres of movie entered by user
        '''
        count1=0
        t=coldata.split(',')
        u=self.user_inputs['genres'].split(',')
        for x in t:
            x1=x.strip()
            for y in u:
                y1=y.strip()
                if x1==y1:
                    count1=count1+1
        return count1
    
    def __keywords_score(self,coldata):
        '''
        The method counts the number of keywords matched with keywords of movie entered by user
        '''
        count1=0
        t=coldata.split(',')
        u=self.user_inputs['new_keywords'].split(',')
        for x in t:
            x1=x.strip()
            for y in u:
                y1=y.strip()
                if x1==y1:
                    count1=count1+1
        
        return count1 
    
 
    def __directors_score(self,coldata):
        '''
        The method counts the number of directors matched with directors of movie entered by user
        '''
        count1=0
        t=coldata.split(',')
        u=self.user_inputs['movie_directors'].split(',')
        for x in t:
            x1=x.strip()
            for y in u:
                y1=y.strip()
                if x1==y1:
                    count1=count1+1
        
        return count1 
     
 
    def __actors_score(self,coldata):
        '''
        The method counts the number of actors matched with actors of movie entered by user
        '''
        count1=0
        t=coldata.split(',')
        u=self.user_inputs['top_movie_cast'].split(',')
        for x in t:
            x1=x.strip()
            for y in u:
                y1=y.strip()
                if x1==y1:
                    count1=count1+1
        
        return count1     
    def __spoken_languages_count(self,coldata):
        '''
        The method counts the number of actors matched with actors of movie entered by user
        '''
        count1=0
        t=coldata.split(',')
        u=self.user_inputs['spoken_languages'].split(',')
        for x in t:
            x1=x.strip()
            for y in u:
                y1=y.strip()
                if x1==y1:
                    count1=count1+1
        
        return count1      
    
       
    def __scoring_moviesData(self):
        #merge_Final_data=movie_data()
        self.merge_Final_data['genres_score'] =self.merge_Final_data['genres'].apply(self.__genres_score)
        self.merge_Final_data['keywords_score'] =self.merge_Final_data['new_keywords'].apply(self.__keywords_score)
        self.merge_Final_data['directors_score'] =self.merge_Final_data['movie_directors'].apply(self.__directors_score)
        self.merge_Final_data['actors_score'] =self.merge_Final_data['top_movie_cast'].apply(self.__actors_score)
        self.merge_Final_data['languages_count'] =self.merge_Final_data['spoken_languages'].apply(self.__actors_score)

        return self.merge_Final_data
    
    def __scaleColumns(self):
        '''
        This method scales genres_score,keywords_score,directors_score,actors_score columns data
        '''
        df=self.__scoring_moviesData()
        cols_to_scale=['genres_score','keywords_score','directors_score','actors_score']
        min_max_scaler = preprocessing.MinMaxScaler()
        for col in cols_to_scale:
            df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(self.__scoring_moviesData()[col])),columns=[col])
        return df
    
    '''
    This method calculates the final score
    '''
    def __final_score(self):
        #x1=scaleColumns(x,cols_to_scale)
        x1=self.__scaleColumns()
        w1=0.3
        w2=0.2
        w3=0.2
        w4=0.3
        x1['final_score']=w1*x1['genres_score']+w2*x1['keywords_score']+w3*x1['directors_score']+w4*x1['actors_score']
        return x1


    def __movieList(self):
        '''
        The method gives top 15 movies on the basis of final score
        '''
        x1=self.__final_score()
        x1=x1[x1['original_title']!=self.userstr]
        x1=x1[x1['languages_count']!=0]
        x1=self.__final_score()
        rec=x1.sort_values(by=['final_score'],ascending=False)
        return rec.head(20)
    
    def top_movies(self):    
        '''
        The method recomends 6 movies to user
        '''
        rec1=self.__movieList()
        rec1=rec1.sort_values(by=['popularity'],ascending=False)
        rec2=rec1['original_title']
        return rec2.head(6) 


class MovieRecommendation():
    def __init__(self):
        dataload_obj=Dataload()
        nlpwordstemming_obj=NlpWordStemming()
        t=dataload_obj.format_Data()
        self.merge_Final_data=nlpwordstemming_obj.movie_data(t)
        self.user_inputs={}
        self.userstr=""
        
    def getmovielist(self):
        ''' it returns names of all the movies '''
        movielist=list()
        for x in self.merge_Final_data['original_title']:
            movielist.append(x)
        return movielist
    
    def __movieRec(self):
        moviescore_obj=MovieScore(self.merge_Final_data,self.user_inputs,self.userstr)
        movies=moviescore_obj.top_movies()
        return movies
   
    def movies_recommendations(self, input_str):
        name=['Either you have entered incorrect movie name or we do not have enough information on given movie to provide you movie recommendation.']
        wel=input_str
        self.userstr=wel.strip().lower()
        desired_row = self.merge_Final_data[self.merge_Final_data['original_title'].str.startswith(self.userstr)]
        if(not desired_row.empty):
            desired_row = desired_row.reset_index()
            self.user_inputs['original_title']=desired_row['original_title'][0]
            self.user_inputs['genres']=desired_row['genres'][0]
            self.user_inputs['movie_id']=desired_row['movie_id'][0]
            self.user_inputs['new_keywords']=desired_row['new_keywords'][0]
            self.user_inputs['movie_directors']=desired_row['movie_directors'][0]
            self.user_inputs['top_movie_cast']=desired_row['top_movie_cast'][0]
            self.user_inputs['spoken_languages']=desired_row['spoken_languages'][0]
            name=self.__movieRec()
        return name
#a=MovieRecommendation()
#print(a.movies_recommendations("thor"))
#print(a.getmovielist())
"""
examples for user input:
titanic
thor
up
avatar
avengers: age of ultron
ocean's eleven
Exorcist: The Beginning
pirates of the caribbean: dead man's chest
rampage
bang
jurassic world
the hobbit
the hobbit: the desolation of smaug

"""