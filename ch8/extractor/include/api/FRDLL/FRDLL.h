#define FRDLL_EXPORTS  
#ifdef FRDLL_EXPORTS  
#define FRDLL_API __declspec(dllexport)   
#else  
#define FRDLL_API __declspec(dllimport)   
#endif  

FRDLL_API int InitAuthenSDKEnvironment(const char* modelPath);

FRDLL_API void freeAuthenSDKEnvironment();

FRDLL_API char *FRGetCode();

FRDLL_API long FRCreateTemplate(const char *pcImageFile, const char *pcTemplateFile);

FRDLL_API long FRTemplateMatch(const char *pcTemplateFileA, const char *pcTemplateFileB, float *pfSim);

FRDLL_API long FRMemoryMatch(const unsigned char *pData1, const unsigned char *pData2, int nLen1, int nLen2, float *pfSim);
