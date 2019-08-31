#include "UrlParseBolt.h"
#include "hurricane/util/StringUtil.h"

#include <iostream>
#include <sstream>
#include <cstring>

#include<sys/types.h>  
#include<sys/socket.h>  
#include<netinet/in.h>  

struct ParsedUrl {
    std::string host;
    std::string path;
};

const int REMOTE_PORT = 8000;

static void RegexGetImages(const string& allHtml, std::vector<std::string>& photoUrls);
static void RegexGetComs(const string &allHtml, std::vector<std::string>& comUrls);

void UrlParseBolt::Prepare(std::shared_ptr<hurricane::collector::OutputCollector> outputCollector) {
    _outputCollector = outputCollector;
    _socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (_socketFd < 0) {
        std::cerr << "Create socket failed" << std::endl;
        return;
    }
}

void UrlParseBolt::Cleanup() {
}

std::vector<std::string> UrlParseBolt::DeclareFields() {
    return{ "word", "count" };
}

void UrlParseBolt::PreConnect(const std::string& url) {
    ParsedUrl parsedUrl;

    if (!ParseUrl(url.c_str(), &parsedUrl)) {
        std::cerr << "Parse url error: " << url << std::endl;
        return;
    }

    struct hostent *hptr = gethostbyname(parsedUrl.host.c_str());
    if(!hptr) {
        std::cerr << "Parse host name error: " << parsedUrl.host << std::endl;
        return;
    }

    struct sockaddr_in serverAddress;
    memset(&servaddr, 0, sizeof(servaddr));  
    serverAddress.sin_family = AF_INET;  
    serverAddress.sin_port = htons(REMOTE_PORT);  
    serverAddress.sin_addr.s_addr = *((unsigned long*)hptr->h_addr_list[0]);

    if (connect(_socketFd,(struct sockaddr *)&servaddr,sizeof(servaddr)) ) {
        std::cerr << "Connect to remote server failed: " << url << std::endl;
        return;
    }

	string reqInfo = "GET " + parsedUrl.path + " HTTP/1.1\r\nHost: " + parsedUrl.host + "\r\nConnection:Close\r\n\r\n";
	if (SOCKET_ERROR == send(_socketFd, reqInfo.c_str(), reqInfo.size(), 0))
	{
		cout << "Send request failed" << endl;
		close(_socketFd);
		return;
	}
}

void UrlParseBolt::Execute(const hurricane::base::Tuple& tuple) {
    std::string url = tuple[0].GetStringValue();
    bool connected = PreConnect(url);
    if (!connected) {
        return;
    }

    std::vector<std::string> photoUrls;
    std::vector<std::string> comUrls;
    PutImageToSet(photoUrls, comUrls);

    for (std::vector<std::string>::iterator it; it != photoUrls.end(); ++ it)
    {
        StoreImage(*it);
    }

    for (std::vector<std::string>::iterator it; it != photoUrls.end(); ++ it)
    {
        std::string newUrl = *it
        _outputCollector->Emit({ newUrl });
    }
}

void UrlParseBolt::PutImageToSet(
        std::vector<std::string>& photoUrls,
        std::vector<std::string>& comUrls)
{
	int n;
	char buf[1024];
    std::string allHtml;
	while ((n = recv(_socketFd, buf, sizeof(buf)-1, 0)) > 0)
	{
		buf[n] = '\0';
		allHtml += string(buf);
	}

	RegexGetImages(allHtml, photoUrls);
	RegexGetComs(allHtml, comUrls);
}

static bool ParseUrl(char *url, ParsedUrl* parsedUrl)
{
    char host[255];
    char othPath[255];

	char *pos = strstr(url, "http://");
	if (pos == NULL)
		return false;
	else
		pos += 7;
	sscanf(pos, "%[^/]%s", host, othPath);   //http:// 后一直到/之前的是主机名

    parsedUrl->host = host;
    parsedUrl->path = othPath;

	return true;
}

void RegexGetImages(const string& allHtml, std::vector<std::string>& photoUrls)
{
	smatch mat;
	regex pattern("src=\"(.*?\.jpg)\"");
	string::const_iterator start = allHtml.begin();
	string::const_iterator end = allHtml.end();
	while (regex_search(start, end, mat, pattern))
	{
		string msg(mat[1].first, mat[1].second);
		photoUrls.push_back(msg);
		start = mat[0].second;
	}
}

void RegexGetComs(const string &allHtml, std::vector<std::string>& comUrls)
{
	smatch mat;
	regex pattern("href=\"(http://[^\s'\"]+)\"");
	string::const_iterator start = allHtml.begin();
	string::const_iterator end = allHtml.end();
	while (regex_search(start, end, mat, pattern))
	{
		string msg(mat[1].first, mat[1].second);
		comUrls.push_back(msg);
		start = mat[0].second;
	}
}

void UrlParseBolt::StoreImage(const string& imageUrl)
{
	int n;
    std::string tempUrl = imageUrl;

	PreConnect(tempUrl);

    std::string photoName;
	photoName.resize(imageUrl.size());
	int k = 0;
	for (int i = 0; i<imageUrl.length(); i++){
		char ch = imageUrl[i];
		if (ch != '\\'&&ch != '/'&&ch != ':'&&ch != '*'&&ch != '?'&&ch != '"'&&ch != '<'&&ch != '>'&&ch != '|')
			photoName[k++] = ch;
	}
	photoName = "./img/"+photoName.substr(0, k) + ".jpg";

    std::fstream file;
	file.open(photoName.c_str(), std::ios_base::out | std::ios_base::binary);
	char buf[1024];
	memset(buf, 0, sizeof(buf));
	n = recv(_socketFd, buf, sizeof(buf)-1, 0);
	char *cpos = strstr(buf, "\r\n\r\n");

	file.write(cpos + strlen("\r\n\r\n"), n - (cpos - buf) - strlen("\r\n\r\n"));
	while ((n = recv(_socketFd, buf, sizeof(buf)-1, 0)) > 0)
	{
		file.write(buf, n);
	}
	file.close();
}
