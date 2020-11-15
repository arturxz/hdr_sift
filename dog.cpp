//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename dog.cpp .cpp` dog.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

const int INF = (int) 1e9;

//Criando imagens do tipo Mat
FILE *in, *out0, *out1, *out2, *out3;

Mat input, inputGray, inputIm[5], octave[4][5], dogI[4][4], response;
Mat roi[4];

bool isHDR = false;

int gaussianSize = 11;

struct KeyPoints{
	int x, y, scale, z;//posicao (x, y) e o octave ou escalada da imagem
	float response;
};

vector<KeyPoints> keyPoint;
vector<pair<int, int> > ROI;
vector<pair<int, int> > quadROIo;
vector<pair<int, int> > quadROIi;


//Pegando maior valor numa imagem cinza
float getMaxValue1(Mat src1){
  float maior = -INF;
  for(int row = 0; row < src1.rows; row++){
	for(int col = 0; col < src1.cols; col++){
		maior = max(maior, src1.at<float>(row, col));
	}
  }
  return maior;
}

uchar getMaxValue2(Mat src1){
  uchar maior = 0;
  for(int row = 0; row < src1.rows; row++){
	for(int col = 0; col < src1.cols; col++){
		maior = max(maior, src1.at<uchar>(row, col));
	}
  }
  return maior;
}
	
//Abrindo imagem no argumento da linha de comando
void read(char *name, char *argv2){
	input = imread(name, IMREAD_UNCHANGED);
	
	//Gerando imagem grayscale
	cvtColor(input, inputGray, COLOR_BGR2GRAY);
	
	//Conferindo se é HDR
	if(input.depth() == CV_32F) {
		isHDR = true;
		normalize(inputGray, inputGray, 0.0, 256.0, NORM_MINMAX, CV_32FC1, Mat());
		printf("Imagem HDR\n");
	}else isHDR = false;
	
	//Lendo arquivo com os pontos (x, y) da ROI
	if(argv2 != NULL){
		string path(argv2); //../dataset/2D/distance/100/100
		string num = "";
		while(true){
			if(path.back() == '/'){
				break;
			}
			num += path.back();
			path.pop_back();
		}
		reverse(num.begin(), num.end());
		
		roi[0] = imread(path+"ROI."+num+".png", IMREAD_UNCHANGED);
		roi[1] = imread(path+"ROIh."+num+".png", IMREAD_UNCHANGED);
		roi[2] = imread(path+"ROIm."+num+".png", IMREAD_UNCHANGED);
		roi[3] = imread(path+"ROIs."+num+".png", IMREAD_UNCHANGED);
		
	}else{
		roi[0] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[1] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[2] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		roi[3] = Mat::zeros(cv::Size(input.cols, input.rows), CV_8U);
		
		for(int x = 0; x < input.cols; x++)
			for(int y = 0; y < input.rows; y++)
				roi[0].at<uchar>(y, x) = 1;
		roi[1] = roi[0];
	}
}

//Função que inicializa os octaves com a mudança de escala e blur para 4 octaves 5 imagens cada (Valor padrão)
void initOctaves(){	
	Mat auxImg;
	inputGray.convertTo(auxImg, CV_32FC1);
	
	float k[] = {0.707107, 1.414214, 2.828428, 5.656856};
	
	for(int c = 0; c < 4; c++){ // escala
		float ko = k[c];
		for(int j = 0; j < 5; j++){			
			GaussianBlur(auxImg, octave[c][j], Size(gaussianSize,gaussianSize), ko, ko, BORDER_DEFAULT);
			ko = ko * 1.414213562; // = ko*sqrt(2);
		}
		//resampling inputImage
		resize(auxImg, auxImg, Size(auxImg.cols/2, auxImg.rows/2));
	}
}

//Função que calula a diferença entre as gaussianas DoG e salva em uma nova imagem
void calcDoG(){
	for(int i = 0; i < 1; i++){
		for(int j = 0; j < 4; j++){
			dogI[i][j] = Mat::zeros(Size(octave[i][j].cols, octave[i][j].rows), CV_32F);
			for(int col = 0; col < octave[i][j].cols; col++){
				for(int row = 0; row < octave[i][j].rows; row++){
					float a = octave[i][j].at<float>(row, col) - octave[i][j+1].at<float>(row, col);
					dogI[i][j].at<float>(row, col) = a;
				}
			}
		}
	}
}

//Conferindo se o valor a ser acessado está dentro dos limites da imagem
bool outOfBounds(int i, int j, Mat input){
	return (i < 0 || j < 0 || i >= input.rows || j >= input.cols);
}

//Selecionando os Keypoinst com a Non Maxima supression 
void nonMaximaSupression(){
	//Selecionando os vizinhos
	int maskSize = 21; // Mascara de 21 x 21 baseado no artigo do prybil
	int cont = 0;
	
	for(int c = 0; c < 1; c++){
		for(int z = 1; z < 3; z++){
			Mat matAux = Mat::zeros(Size(dogI[c][z].cols, dogI[c][z].rows), CV_32F);
			for(int x = 50; x < dogI[c][z].cols-50; x++){
				for(int y = 50; y < dogI[c][z].rows-61; y++){
					float mid = dogI[c][z].at<float>(y, x);	
					
					if(roi[0].at<uchar>(y, x) == 0){
						matAux.at<float>(y, x) = 0;
						continue;
					}
					
					bool cond1 = true, cond2 = true;
					int mSize2 = maskSize/2;
					
					for(int i = y - mSize2; i <= y + mSize2; i++){
						for(int j = x - mSize2; j <= x + mSize2; j++){
							if(!outOfBounds(i, j, dogI[c][z])){
								if(!((mid < dogI[c][z].at<float>(i, j) || (y == i && x == j)) && 
								   mid < dogI[c][z-1].at<float>(i, j) && 
								   mid < dogI[c][z+1].at<float>(i, j))
								  )
								 {
									 cond1 = false;
									 break;
								 }
							}
						}
						if(!cond1)break;
					}
					for(int i = y - mSize2; i <= y + mSize2; i++){
						for(int j = x - mSize2; j <= x + mSize2; j++){
							if(!outOfBounds(i, j, dogI[c][z])){
								if(!((mid > dogI[c][z].at<float>(i, j) || (y == i && x == j)) && 
								   mid > dogI[c][z-1].at<float>(i, j) && 
								   mid > dogI[c][z+1].at<float>(i, j))
								  )
								 {
									 cond2 = false;
									 break;
								 }
							}
						}
						if(!cond2)break;
					}
					if(cond1 || cond2){
						matAux.at<float>(y, x) = dogI[c][z].at<float>(y, x);
						cont++;
					}else matAux.at<float>(y, x) = 0;
				}
			}
			dogI[c][z] = matAux;
		}
		cout<<cont<<endl;//Mostrando quantidade de keypoints encontrado na imagem i j
		cont = 0;
	}
}//Fim função

//Passando o Limiar na imagem resultante response
void threshold(float val){
	//Atualizando threshold
	float thresholdValue = val;
	int cont = 0;
	int begX = 0, begY = 0; 
	int endX = inputGray.cols, endY = inputGray.rows;
	
	for(int c = 0; c < 1; c++){ //scale/octave
		for(int z = 1; z < 3; z++){
			for(int y = begY; y < endY-10; y++){
				for(int x = begX; x < endX-10; x++){
						
					float val = fabs(dogI[c][z].at<float>(y, x));
					if(val > 0){
						keyPoint.push_back({x, y, c, z,val});
						cont++;
					}else{
						 dogI[c][z].at<float>(y, x) = 0;
					}
				}
			}
		}
		cout<<"t: "<<cont<<endl;
		cont = 0;
			
		begX /= 2; endX /= 2;
		begY /= 2; endY /=2;
	}
}

void saveKeypoints(){
	printf("Salvando keypoints ROIs no arquivo...\n");
	
	vector<pair<float, pair<int, int> > > aux1, aux2, aux3;
	vector<pair<float, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++){
		int y = keyPoint[i].y, x = keyPoint[i].x;
		aux.push_back({-keyPoint[i].response, {y, x}});
	}
    sort(aux.begin(), aux.end());
    
    int quantMaxKP = 500;
    //int quantMaxKP = 5400;
    
    for(int i = 0; i < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	
	 	if(y >= roi[1].rows || x >= roi[1].cols) continue;
	 	if(y >= roi[2].rows || x >= roi[2].cols) continue;
	 	if(y >= roi[3].rows || x >= roi[3].cols) continue;
	 	
	 	if(roi[1].at<uchar>(y, x) != 0) aux1.push_back({aux[i].first, {y, x}});
	 	else if(roi[2].at<uchar>(y, x) != 0) aux2.push_back({aux[i].first, {y, x}});
	 	else if(roi[3].at<uchar>(y, x) != 0) aux3.push_back({aux[i].first, {y, x}});
    }
    
    if(aux3.size() > 500) aux3.clear();
    
    double T = aux1.size() + aux2.size() + aux3.size();
	
	double minFp = min(aux1.size()/T, min(aux2.size()/T, aux3.size()/T));
	double maxFp = max(aux1.size()/T, max(aux2.size()/T, aux3.size()/T));
	double D = 1 - (maxFp - minFp);
	
	//Salvando Distribution Rate
	//fprintf(out0, "%.4f\n", D);
    
	keyPoint.clear();
	for(int i = 0; i < aux1.size(); i++)
		keyPoint.push_back({aux1[i].second.first, aux1[i].second.second});
	for(int i = 0; i < aux2.size(); i++)
		keyPoint.push_back({aux2[i].second.first, aux2[i].second.second});
	for(int i = 0; i < aux3.size(); i++)
		keyPoint.push_back({aux3[i].second.first, aux3[i].second.second});
	
	//Salvando pontos ROI 1
	fprintf(out1, "%d\n", (int)aux1.size());
	for(int i = 0; i < (int)aux1.size(); i++)
		fprintf(out1, "%d %d %.4f\n", aux1[i].second.first, aux1[i].second.second, -aux1[i].first);
	fclose(out1);
	
	//Salvando pontos ROI 2
	fprintf(out2, "%d\n", (int)aux2.size());
	for(int i = 0; i < (int)aux2.size(); i++)
		fprintf(out2, "%d %d %.4f\n", aux2[i].second.first, aux2[i].second.second, -aux2[i].first);
	fclose(out2);
	
	//Salvando pontos ROI 3
	fprintf(out3, "%d\n", (int)aux3.size());
	for(int i = 0; i < (int)aux3.size(); i++)
		fprintf(out3, "%d %d %.4f\n", aux3[i].second.first, aux3[i].second.second, -aux3[i].first);
	fclose(out3);
}
void saveKeypoints2ROIs(){
	printf("Salvando keypoints 2 ROIs no arquivo...\n");
	
	vector<pair<float, pair<int, int> > > aux1, aux2, aux3;
	vector<pair<float, pair<int, int> > > aux;
	
    for(int i = 0; i < (int)keyPoint.size(); i++){
		int y = keyPoint[i].y, x = keyPoint[i].x;
		aux.push_back({-keyPoint[i].response, {y, x}});
	}
    sort(aux.begin(), aux.end());
    
    int quantMaxKP = 500;
    
    for(int i = 0; i < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	
	 	if(y >= roi[1].rows || x >= roi[1].cols) continue;
	 	if(y >= roi[3].rows || x >= roi[3].cols) continue;
	 	
	 	if(roi[1].at<uchar>(y, x) != 0) aux1.push_back({aux[i].first, {y, x}});
	 	else if(roi[3].at<uchar>(y, x) != 0) aux3.push_back({aux[i].first, {y, x}});
    }
    
    if(aux3.size() > 500) aux3.clear();
    
    double T = aux1.size() + aux3.size();
	
	double minFp = min(aux1.size()/T, aux3.size()/T);
	double maxFp = max(aux1.size()/T, aux3.size()/T);
	double D = 1 - (maxFp - minFp);
	
	//Salvando Distribution Rate
	//fprintf(out0, "%.4f\n", D);
    
	keyPoint.clear();
	for(int i = 0; i < aux1.size(); i++)
		keyPoint.push_back({aux1[i].second.first, aux1[i].second.second});
	for(int i = 0; i < aux3.size(); i++)
		keyPoint.push_back({aux3[i].second.first, aux3[i].second.second});
	
	//Salvando pontos ROI 1
	fprintf(out1, "%d\n", (int)aux1.size());
	for(int i = 0; i < (int)aux1.size(); i++)
		fprintf(out1, "%d %d %.4f\n", aux1[i].second.first, aux1[i].second.second, -aux1[i].first);
	fclose(out1);
	//Salvando pontos ROI 3
	fprintf(out3, "%d\n", (int)aux3.size());
	for(int i = 0; i < (int)aux3.size(); i++)
		fprintf(out3, "%d %d %.4f\n", aux3[i].second.first, aux3[i].second.second, -aux3[i].first);
	fclose(out3);
}


void showKeyPoints(){
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].y;
		int y = keyPoint[i].x;

		circle(input, Point (x, y), 1, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (x, y), 3, Scalar(0, 0, 255), 1, 8, 0);
		circle(input, Point (x, y), 4, Scalar(0, 255, 0), 1, 8, 0);
		
	}
}

//Função Principal
int main(int, char** argv ){
	char saida[255];
	strcpy(saida, argv[1]);
	saida[strlen(saida)-4] = '\0';
	string saida2(saida);
	
	string saida1 = saida2;
	string saida3 = saida2;
	string saida4 = saida3;
	
	saida1 += ".dog.distribution.txt";
	saida2 += ".dog1.txt";
	saida3 += ".dog2.txt";
	saida4 += ".dog3.txt";
	
	//out0 = fopen(saida1.c_str(), "w+");
	out1 = fopen(saida2.c_str(), "w+");
	out2 = fopen(saida3.c_str(), "w+");
	out3 = fopen(saida4.c_str(), "w+");
	
	read(argv[1], argv[2]);
	
	//Inicializando octaves
	initOctaves();
	
	//Calculando DoG
	calcDoG();
	
	Mat responseEx = dogI[0][1].mul(45);
	
	/*
	for(int y = 0; y < response.rows; y++){
		for(int x = 0; x < response.cols; x++){
			reponse.at<float>(y, x) *= 2;
		}
	}
	*/
	
	//imwrite("response.png", responseEx);

	//Fazendo NonMaximaSupression na imagem DoG
	nonMaximaSupression();
	
	//Limiar na imagem de Response
	threshold(8); // Threshold fixo para teste do pribyl = 8
		
	//Limiar p/ edges response
	//edgeThreshold(); 
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	saveKeypoints();
	//saveKeypoints2ROIs();
	
	showKeyPoints();
	
	//Salvando imagens com Keypoints
	int len = strlen(saida);
	saida[len] = 'R';saida[len+1] = '.';saida[len+2] = 'p';saida[len+3] = 'n';saida[len+4] = 'g';saida[len+5] = '\0';
	imwrite(saida, input);
	
	cout<<"done!\n";
	
	//Mostrando imagem com Keypoints
	//imshow("Result", input);
	//waitKey(0);
	
	return 0;
}
