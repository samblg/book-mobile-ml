/search/BUILD:
cc_binary(name = ‘google_search_page’,
       deps = [ ‘:search’,
                  ‘:show_results’])

cc_library(name = ‘search’,
       srcs = [ ‘search.h’,‘search.cc’],
       deps = [‘//index:query’])

/index/BUILD:

cc_library(name = ‘query’,
       srcs = [ ‘query.h’, ‘query.cc’, ‘query_util.cc’],
       deps = [‘:ranking’,
               ‘:index’])

cc_library(name = ‘ranking’,
       srcs = [‘ranking.h’, ‘ranking.cc’],
       deps = [‘:index’, 
               ‘//storage/database:query’])

cc_library(name = ‘index’,
       srcs = [‘index.h’, ‘index.cc’],
       deps = [‘//storage/database:query’]) 
