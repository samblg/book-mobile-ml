#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <sys/stat.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>

using namespace std;

typedef struct {
    int width;
    int height;
    float* elements;
    int stride;
} Matrix;

class dataImage {
public:
    unsigned char pixArray[28 * 28];
    unsigned char label[1];
    
    friend std::ostream & operator<<(std::ostream & _stream, dataImage const & dataImg) {
        _stream << "Label: " << unsigned(dataImg.label[0]) << endl;
        _stream << "Image: " << endl;
        for (int x = 0; x < 28*28; x++) {
            _stream << setfill('0') << setw(1) << (int) dataImg.pixArray[x] << " ";
            if (x % 28 == 0)
                _stream << endl;
            
        }
        return _stream;
    }
};

class neuron {
public:
    
    float bias = 0.0;
    vector<float> weights;
    float dk;
    vector<float> nudges;
    float val  = 0;
    
    neuron() : weights(vector<float>()) {}
    neuron(const int layerSize) :  weights(vector<float>(layerSize,0.01)),nudges(vector<float>(layerSize, 0)) {}
    ~neuron() {
        weights.empty();
    }
};

class database {
private:
    int images_magic_number;
    int labels_magic_number;
    int number_of_images;
    int number_of_labels;
    int nrows;
    int ncolumns;
    
public:
    
    vector<dataImage*> *dataset;
    
    database(const string imageDBfilename, const string labelDBfilename) {
        bool dbgFunction = false;
        std::vector<unsigned char> *image_buffer;
        
        ifstream image_file(imageDBfilename, std::ios::binary | std::ios::ate | std::ios::in);
        
        
        if (image_file.is_open()) {
            std::streamsize size = image_file.tellg();
            image_file.seekg(0, std::ios::beg);
            
            image_buffer = new std::vector<unsigned char>(size);
            if (image_file.read((char*)image_buffer->data(), size))
            {
                if (dbgFunction)
                    cout << "Opened and loaded Image Dataset." << endl;
            }
            image_file.close();
            if (dbgFunction)
                cout << "Image Dataset Closed." << endl;
        }
        else {
            cout << "Unable to open Image Dataset file." << endl;
            exit(0);
        }
        if (dbgFunction)
            cout << "Building training database. . ." << endl;
        
        images_magic_number = int(
                                  (unsigned char)(image_buffer->at(0)) << 24 |
                                  (unsigned char)(image_buffer->at(1)) << 16 |
                                  (unsigned char)(image_buffer->at(2)) << 8 |
                                  (unsigned char)(image_buffer->at(3))
                                  );
        
        number_of_images = int(
                               (unsigned char)(image_buffer->at(4)) << 24 |
                               (unsigned char)(image_buffer->at(5)) << 16 |
                               (unsigned char)(image_buffer->at(6)) << 8 |
                               (unsigned char)(image_buffer->at(7))
                               );
        
        nrows = int(
                    (unsigned char)(image_buffer->at(8)) << 24 |
                    (unsigned char)(image_buffer->at(9)) << 16 |
                    (unsigned char)(image_buffer->at(10)) << 8 |
                    (unsigned char)(image_buffer->at(11))
                    );
        
        ncolumns = int(
                       (unsigned char)(image_buffer->at(12)) << 24 |
                       (unsigned char)(image_buffer->at(13)) << 16 |
                       (unsigned char)(image_buffer->at(14)) << 8 |
                       (unsigned char)(image_buffer->at(15))
                       );
        
        dataset = new vector<dataImage*>(number_of_images);
        
        if (dbgFunction)
            cout << "Total images: " << number_of_images << endl;
        
        int memblockCursor = 16;
        
        for (int i = 0; i < number_of_images; i++) {
            dataImage *nImage = new dataImage();
            for (int row = 0; row < nrows*nrows; row++) {
                nImage->pixArray[row] = image_buffer->at(memblockCursor++);
            }
            dataset->at(i) = nImage;
        }
        if (dbgFunction)
            cout << "Releasing Image File Buffer." << endl;
        delete image_buffer;
        
        ifstream label_file(labelDBfilename, std::ios::binary | std::ios::ate);
        
        std::vector<char> *label_buffer;
        
        if (label_file.is_open()) {
            std::streamsize size = label_file.tellg();
            label_file.seekg(0, std::ios::beg);
            
            label_buffer = new std::vector<char>(size);
            if (label_file.read(label_buffer->data(), size))
            {
                if (dbgFunction)
                    cout << "Opened and loaded Label Dataset." << endl;
            }
            label_file.close();
            if (dbgFunction)
                cout << "Label Dataset Closed." << endl;
        }
        else {
            cout << "Unable to open Label Dataset file." << endl;
            exit(0);
        }
        
        labels_magic_number = int(
                                  (unsigned char)(label_buffer->at(0)) << 24 |
                                  (unsigned char)(label_buffer->at(1)) << 16 |
                                  (unsigned char)(label_buffer->at(2)) << 8 |
                                  (unsigned char)(label_buffer->at(3))
                                  );
        
        number_of_labels = int(
                               (unsigned char)(label_buffer->at(4)) << 24 |
                               (unsigned char)(label_buffer->at(5)) << 16 |
                               (unsigned char)(label_buffer->at(6)) << 8 |
                               (unsigned char)(label_buffer->at(7))
                               );
        
        
        memblockCursor = 8; //Point to the begining of our first image
        
        for (int i = 0; i < number_of_labels; i++) {
            dataImage *nImage = dataset->at(i);
            nImage->label[0] = (unsigned char) label_buffer->at(memblockCursor++);
        }
        if (dbgFunction)
            cout << "Releasing Label File Buffer." << endl;
        delete label_buffer;
        
        if (dbgFunction)
            cout << "Built training database!" << endl;
    }
    
    ~database() {
        cout << "! Deleting Training Database from RAM !" << endl;;
        for (dataImage *x : *dataset) {
            delete x;
        }
        delete dataset;
    }
    
};

float sigmoid(float num) {
    bool useTrueSigmoid = true;
    if (useTrueSigmoid) {
        float exp_val = exp(-num);
        return (float) (1 / (float) (1 + exp_val));
    }
    else
        return (num / (1 + abs(num)));
}

float sigprime(float num) {
    return sigmoid(num) * (1 - sigmoid(num));
}

class neuralLayer {
public:
    vector<neuron> listNeurons;
    
    neuralLayer(vector<neuron> input) : listNeurons(input){}
    
    void randomizeNeuronWeights(bool andBias) {
        for (int i = 0; i < listNeurons.size(); i++) {
            for (int x = 0; x < listNeurons.at(i).weights.size(); x++) {
                listNeurons.at(i).weights.at(x) = (float)((rand() % 20) - 10) / 10;
            }
            if (andBias)
                listNeurons.at(i).bias = (float)((rand() % 20) - 10);
        }
    }
};

void multLoopLayers(neuralLayer* valLayer, neuralLayer *weightLayer) {
    for (int i = 0; i < weightLayer->listNeurons.size(); i++) {
        float sum = 0;
        for (int j = 0; j < valLayer->listNeurons.size(); j++) {
            float w, a;
            a = valLayer->listNeurons.at(j).val;
            w = weightLayer->listNeurons.at(i).weights.at(j);
            
            sum += w * a;
            
        }
        
        weightLayer->listNeurons.at(i).val = sigmoid(sum + weightLayer->listNeurons.at(i).bias);
        
    }
}

float calculateCost(vector<neuron>* outputLayer, unsigned char* label) {
    float difference = 0;
    for (int i = 0; i < outputLayer->size(); i++) {
        if (i == *label) {
            difference += pow(outputLayer->at(i).val - 1, 2);
        }
        else {
            difference += pow(outputLayer->at(i).val, 2);
        }
    }
    return difference / 2;
}


long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void loadWeights(neuralLayer *A, neuralLayer *B) {
    ifstream myLayer1file ("neuralNetValuesLayer1.txt");
    if (myLayer1file.is_open())
    {
        
        myLayer1file.seekg(0, std::ios::beg);
        char temp[10240 / 2];
        myLayer1file.getline(temp, 10);
        
        std::streamsize size = GetFileSize("neuralNetValuesLayer1.txt") - myLayer1file.tellg();
        
        vector<unsigned char> *fileBuf = new std::vector<unsigned char>(size);
        if (myLayer1file.read((char*)fileBuf->data(), size))
        {
        }
        char delimiter = '@';
        int index = 0;
        
        std::string s(fileBuf->begin(), fileBuf->end());
        
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            
            string l = string(token);
            size_t pos2 = 0;
            std::string toke;
            int k = 0;
            while ((pos2 = l.find('|')) != std::string::npos) {
                toke = l.substr(0, pos2);
                float o = atof(toke.c_str());
                //cout << l << std::endl;
                A->listNeurons.at(index).weights.at(k) = o;
                l.erase(0, pos2 + 1);
                k++;
            }
            s.erase(0, pos + 1);
            index++;
        }
        
        myLayer1file.close();
        
    } else cout << "Unable to open file";
    
    ifstream myLayer2file("neuralNetValuesLayer2.txt");
    if (myLayer2file.is_open())
    {
        
        myLayer2file.seekg(0, std::ios::beg);
        
        char temp[10240 / 2];
        
        myLayer2file.getline(temp, 10);
        
        std::streamsize size = GetFileSize("neuralNetValuesLayer2.txt") - myLayer2file.tellg();
        
        vector<unsigned char> *fileBuf = new std::vector<unsigned char>(size);
        if (myLayer2file.read((char*)fileBuf->data(), size))
        {
        }
        char delimiter = '@';
        int index = 0;
        
        std::string s(fileBuf->begin(), fileBuf->end());
        
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            
            string l = string(token);
            size_t pos2 = 0;
            std::string toke;
            int k = 0;
            while ((pos2 = l.find('|')) != std::string::npos) {
                toke = l.substr(0, pos2);
                float o = atof(toke.c_str());
                //cout << l << std::endl;
                B->listNeurons.at(index).weights.at(k) = o;
                l.erase(0, pos2 + 1);
                k++;
            }
            s.erase(0, pos + 1);
            index++;
        }
        
        myLayer2file.close();
        
    }
    else cout << "Unable to open file";
}

int main() {
    database x("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    
    cout << "Images in database: " << x.dataset->size() << endl;
    
    const int layer0_size = 28 * 28;
    const int layer1_size = 500;
    const int layer2_size = 5;
    
    neuralLayer layer1 = neuralLayer(vector<neuron>(layer1_size, neuron(layer0_size)));
    layer1.randomizeNeuronWeights(false);
    
    neuralLayer layer2 = neuralLayer(vector<neuron>(layer2_size, neuron(layer1_size)));
    loadWeights(&layer1, &layer2);
    
    int successfullGuesses = 0;
    int totalGuesses = 0;
    
    for (int r = 0; r < 1; r++) {
        for (int index = 0; index < 1000; index++) {
            dataImage *nImage = x.dataset->at(index);
            if (*nImage->label >= 5) {
                continue;
            }
            neuralLayer layer0 = neuralLayer(vector<neuron>(28 * 28));
            
            for (int i = 0; i < layer0_size; i++)
                layer0.listNeurons.at(i).val = nImage->pixArray[i];
            
            multLoopLayers(&layer0, &layer1);
            multLoopLayers(&layer1, &layer2);
            
            float Cost = calculateCost(&layer2.listNeurons, nImage->label);
            
            for (int i = 0; i < layer2.listNeurons.size(); i++) {
                neuron *layer2Neuron = &layer2.listNeurons.at(i);
                for (int x = 0; x < layer2Neuron->weights.size(); x++) {
                    
                    float delta = 0.0;
                    float Ok = layer2.listNeurons.at(i).val;
                    if (i == *nImage->label) {
                        delta = (Ok - 1) * ((1 - Ok) * Ok)  * layer1.listNeurons.at(x).val;
                        layer2.listNeurons.at(i).dk = (Ok - 1) * ((1 - Ok) * Ok);
                    }
                    else {
                        delta = Ok * ((1 - Ok) * Ok) *  layer1.listNeurons.at(x).val;
                        layer2.listNeurons.at(i).dk = Ok * ((1 - Ok) * Ok);
                    }
                    layer2.listNeurons.at(i).nudges.at(x) = delta;
                }
            }
            
            for (int i = 0; i < layer1.listNeurons.size(); i++) {
                for (int j = 0; j < layer1.listNeurons.at(i).weights.size(); j++) {
                    float Oi = layer0.listNeurons.at(j).val;
                    float Oj = layer1.listNeurons.at(i).val;
                    
                    float delta = Oi * Oj * (1 - Oj);
                    float prod = 0.0;
                    for (int x = 0; x < layer2.listNeurons.size(); x++) {
                        prod += layer2.listNeurons.at(x).dk * layer2.listNeurons.at(x).weights.at(i);
                    }
                    layer1.listNeurons.at(i).weights.at(j) += 0.5 * -delta * prod;
                }
            }
            
            for (int i = 0; i < layer2.listNeurons.size(); i++) {
                neuron *layer2Neuron = &layer2.listNeurons.at(i);
                for (int x = 0; x < layer2Neuron->weights.size(); x++) {
                    layer2.listNeurons.at(i).weights.at(x) += 0.5 * -layer2.listNeurons.at(i).nudges.at(x);
                }
            }
            
            int maxIndex = 0;
            for (int indx = 0; indx < layer2.listNeurons.size(); indx++) {
                if (layer2.listNeurons.at(indx).val > layer2.listNeurons.at(maxIndex).val)
                    maxIndex = indx;
            }
            totalGuesses++;
            
            cout << "Index: " << index << "\tLabel: " << (int)nImage->label[0] << "\tCost: " << setw(10) << setprecision(6) << Cost << "\t";
            
            if (maxIndex == *nImage->label) {
                successfullGuesses++;
                
                cout << "Accuracy: ";
                
                cout << fixed << setw(9) << (((float)successfullGuesses / totalGuesses) * 100) << "%\t";
                
                cout << "CORRECT GUESS!" << endl;
            } else {
                cout << "Accuracy: ";
                
                cout << fixed << setw(9) << (((float)successfullGuesses / totalGuesses) * 100) << "%\t";
                
                cout << endl;
            }
        }
    }
    
    cout << "Done" << endl;
    
    ofstream myLayer1file("neuralNetValuesLayer1.txt");
    if (myLayer1file.is_open())
    {
        myLayer1file << layer1.listNeurons.size() << endl;
        for (int i = 0; i < layer1.listNeurons.size(); i++) {
            for (int x = 0; x < layer1.listNeurons.at(i).weights.size(); x++) {
                myLayer1file << std::setprecision(32) << (float)layer1.listNeurons.at(i).weights.at(x) << ((x == layer1.listNeurons.at(i).weights.size() - 1) ? "|" : "|");
            }
            myLayer1file << "@";
        }
        myLayer1file.close();
    }
    else cout << "Unable to open file";
    
    ofstream myLayer2file("neuralNetValuesLayer2.txt");
    if (myLayer2file.is_open())
    {
        myLayer2file << layer2.listNeurons.size() << endl;
        for (int i = 0; i < layer2.listNeurons.size(); i++) {
            for (int x = 0; x < layer2.listNeurons.at(i).weights.size(); x++) {
                myLayer2file << std::setprecision(32) << (float)layer2.listNeurons.at(i).weights.at(x) << ((x == layer2.listNeurons.at(i).weights.size() - 1) ? "|" : "|");
            }
            myLayer2file << "@";
        }
        
        myLayer2file.close();
    }
    else cout << "Unable to open file";
    
    return 0;
}
