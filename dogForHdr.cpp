//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename dogForHdr.cpp .cpp` dogForHdr.cpp `pkg-config --libs opencv`

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//Incluindo todas as bibliotecas std do c/c++
#include <bits/stdc++.h>
#include "dog.hpp"

using namespace cv;
using namespace std;

const int INF = (int) 1e9;

//Criando imagens do tipo Mat
FILE *in, *out0, *out1, *out2, *out3;

Mat input, inputGray, inputIm[5], octave[4][5], dogI[4][4], dogCopy[5];
Mat roi[4], Ix, Iy, Ix2, Iy2, Ixy;

bool isHDR = false;

double gKernel[5][5];

float mediaGeral = 0; //Media geral dos coefV

int gaussianSize = 11;

vector<KeyPoints> keyPoint;

//Pegando maior valor numa imagem float cinza
float getMaxValue(Mat src1, int xBeg, int xEnd,int yBeg, int yEnd){
  float maior = -1e22;
  for(int y = yBeg; y < yEnd; y++){
	for(int x = xBeg; x < xEnd; x++){
		maior = max(maior, src1.at<float>(y, x));
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
	}
}
//função para aplicar a tranformação logaritmica na image HDR
//Parametros: c constante de multiplicacao da formula
Mat logTranformUchar(Mat src, int c){
	for(int y = 0; y < src.rows; y++){
		for(int x = 0; x < src.cols; x++){
			float r = src.at<uchar>(y, x);
			float val = c * log10(r + 1);
			src.at<uchar>(y, x) = val;
		}
	}
	return src;
}

//Coeficiente de Variacao
Mat coefficienceOfVariationMask(Mat aux){
	
	if(aux.depth() != CV_32F)
		aux.convertTo(aux, CV_32F);
	
	Mat response = aux;
	
	Mat auxResponse = Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
	int n = 3;//maskSize impar
	int N = n*n, cont = 0;//quantidade de pixels visitados
	
	Mat response2 = Mat::zeros(cv::Size(response.cols, response.rows), CV_64F);
	//response * response 
	for(int y = 0; y < response.rows; y++)
		for(int x = 0; x < response.cols; x++)
			response2.at<float>(y, x) = (response.at<float>(y, x) * response.at<float>(y, x));
	
	float sum1 = 0, sum2 = 0;
	
	for(int y = 1; y < n; y++){
		for(int x = 0; x <= n; x++){
			sum1 += response.at<float>(y, x);
			sum2 += response2.at<float>(y, x);
		}
	}
	
	//"Convolution"
	for(int i = (n/2)+1; i < response.rows - (n/2); i++){
		int yBeg = i-(n/2), yEnd = i+(n/2);
		for(int j = (n/2); j < response.cols - (n/2); j++){
			//passando mascara 
			float sumVal = 0, sumVal2 = 0, maior = 0;
			int xBeg = j-(n/2), xEnd = j+(n/2);
			
			for(int y = yBeg; y <= yEnd; y++){
				for(int x = xBeg; x <= xEnd; x++){
					sumVal += response.at<float>(y, x);
					sumVal2 += response2.at<float>(y, x);
					maior = max(maior, response.at<float>(y, x));
				}
			}
						
			float media = sumVal/N;
			
			float variancia = (sumVal2/N) - (media*media);

			float S = sqrt(variancia); // desvio padrao
			float CV = media == 0? 0 : S/media; // Coef de Variacao
			auxResponse.at<float>(i, j) = CV * 100;
			
			//printf("%d %d %f\n", i, j, CV);
			//printf("%.8f %.8f %.8f %.8f %.8f %.8f\n", sumVal, sumVal2, media, variancia, S, CV);
			
			mediaGeral += CV;	
		}
	}

	mediaGeral = mediaGeral/((response.cols-n)*(response.rows-n));
	printf("Media do Coefv = %.10f\n", mediaGeral);//
	
	//Response recebe o valor de coef salvo em aux
	response = auxResponse;
	
	Mat aux2;
	normalize(response, aux2, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	
	return aux2;
}

//Criando kernel para o filtro Gaussian
void createFilter(double sigma){
    double r, s = 2.0 * sigma * sigma;
 
    // sum is for normalization
    double sum = 0.0;
 
    // generate 5x5 kernel
    for (int x = -2; x <= 2; x++)
    {
        for(int y = -2; y <= 2; y++)
        {
            r = sqrt(x*x + y*y);
            gKernel[x + 2][y + 2] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + 2][y + 2];
        }
    }
 
    // normalize the Kernel
    for(int i = 0; i < 5; ++i)
        for(int j = 0; j < 5; ++j)
            gKernel[i][j] /= sum;
 
}

//Gaussian Blur function (input image, gaussian blur mask size
Mat gaussianBlurHDR(Mat input, int siz){
	int n = siz;
	Mat aux = Mat::zeros(cv::Size(input.cols, input.rows), CV_32F);
	for(int i = (n/2)+1; i < input.rows - (n/2); i++){
		int yBeg = i-(n/2), yEnd = i+(n/2);
		for(int j = (n/2); j < input.cols - (n/2); j++){
			int xBeg = j-(n/2), xEnd = j+(n/2);			
			float sum = 0;
			for(int maskI = 0, y = yBeg; y <= yEnd; y++, maskI++){
				for(int maskJ = 0, x = xBeg; x <= xEnd; x++, maskJ++){
					sum += (input.at<float>(y, x) * gKernel[maskI][maskJ]);
				}
			}
			aux.at<float>(i, j) = sum;
		}
	}
	return aux;
}

//Função que inicializa os octaves com a mudança de escala e blur para 4 octaves 5 imagens cada (Valor padrão)
void initOctaves(){		
	//Parte adicional para o HDR 
	GaussianBlur(inputGray, inputGray, Size(5, 5), 0, 0, BORDER_DEFAULT);
	inputGray = coefficienceOfVariationMask(inputGray);	
	
	//imwrite("in.jpg", inputGray);
	
	inputGray = logTranformUchar(inputGray, 2);//Tranformaçao logarítimica com constante c = 
	
	
	Mat auxImg;
	inputGray.convertTo(auxImg, CV_32FC1);
	
	float k[] = {0.707107, 1.414214, 2.828428, 5.656856};
	
	for(int c = 0; c < 4; c++){ //escala
		float ko = k[c];
		for(int j = 0; j < 5; j++){
			GaussianBlur(auxImg, octave[c][j], Size(21,21), ko, ko, BORDER_DEFAULT);
			//createFilter(ko);//Criando mascara para a Gaussiana
			//octave[c][j] = gaussianBlurHDR(auxImg, 5);
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
			//imshow("in", dogI[i][j]); waitKey(0);		
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
			dogCopy[z] = dogI[c][z];
			Mat matAux = Mat::zeros(Size(dogI[c][z].cols, dogI[c][z].rows), CV_32F);
			for(int x = 0; x < dogI[c][z].cols-61; x++){
				for(int y = 0; y < dogI[c][z].rows-61; y++){
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
void threshold(){
	int cont = 0;
	int begX = 0, begY = 0; 
	int endX = inputGray.cols, endY = inputGray.rows;
	
	for(int c = 0; c < 1; c++){ //scale/octave
		for(int z = 1; z < 3; z++){
			
			float maior = getMaxValue(dogI[c][z], begX, endX, begY, endY);
			
			cout<<"maior : "<<maior<<endl;
			
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

void showKeyPoints(){
	for(int i = 0; i < (int)keyPoint.size(); i++){
		int x = keyPoint[i].y;
		int y = keyPoint[i].x;

		circle(input, Point (x, y), 4, Scalar(0, 0, 255), 1, 8, 0);
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
    
    int quantMaxKP = 15000;
    
    for(int i = 0; i < quantMaxKP && i < aux.size(); i++){
	 	int y = aux[i].second.first, x = aux[i].second.second;
	 	if(y >= roi[1].rows || x >= roi[1].cols) continue;
	 	
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
	fprintf(out0, "%.4f\n", D);
    
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
	 	
	 	if(roi[1].at<uchar>(y, x) != 0) aux1.push_back({aux[i].first, {y, x}});
	 	else if(roi[3].at<uchar>(y, x) != 0) aux3.push_back({aux[i].first, {y, x}});
    }
    
    if(aux3.size() > 500) aux3.clear();
    
    double T = aux1.size() + aux3.size();
	
	double minFp = min(aux1.size()/T,  aux3.size()/T);
	double maxFp = max(aux1.size()/T, aux3.size()/T);
	double D = 1 - (maxFp - minFp);
	
	//Salvando Distribution Rate
	fprintf(out0, "%.4f\n", D);
    
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

//edgeThreshold

void edgeThreshold(){
	for(int z = 1; z < 3; z++){
		//Computando sobel operator (derivada da gaussiana) no eixo x e y
		Sobel(dogCopy[z], Ix, CV_32F, 1, 0, 7, 1, 0, BORDER_DEFAULT);
		Sobel(dogCopy[z], Iy, CV_32F, 0, 1, 7, 1, 0, BORDER_DEFAULT);
		
		Ix2 = Ix.mul(Ix); // Ix^2
		Iy2 = Iy.mul(Iy);// Iy^2
		Ixy = Ix.mul(Iy);// Ix * Iy
		
		Mat response = Mat::zeros(cv::Size(dogCopy[z].cols, dogCopy[z].rows), CV_32F);
		for(int row = 0; row < dogCopy[z].rows; row++){
			for(int col = 0; col < dogCopy[z].cols; col++){
				float fx2 = Ix2.at<float>(row, col);
				float fy2 = Iy2.at<float>(row, col);
				float fxy = Ixy.at<float>(row, col);
				float det = (fx2 * fy2) - (fxy * fxy);
				float trace = (fx2 + fy2);
				response.at<float>(row, col) = det - 0.04*(trace*trace);
				//if(response.at<float>(row, col) < 0) response.at<float>(row, col) = 0;
				//printf("%f", response.at<float>(row, col));
			}
		}
		
		for(int row = 0; row < dogCopy[z].rows; row++){
			for(int col = 0; col < dogCopy[z].cols; col++){
				float val = response.at<float>(row, col);
				if(val < -1e4){
					response.at<float>(row, col) = 0;
				}
			}	
		}
		
		vector<KeyPoints> auxKp;
		for(int i = 0; i < (int)keyPoint.size(); i++){
			int y = keyPoint[i].x, x = keyPoint[i].y;
			if(response.at<float>(y, x) != 0)
				auxKp.push_back(keyPoint[i]);
		}
		keyPoint = auxKp;
		
		cout<<"edgeThr end\n";			
	}

}

//Função Principal
// ROI = Region Of Interest
//Chamada: ./siftForHdr Imagem ArquivoROIPoints
int main(int, char** argv ){
	char saida[255];
	strcpy(saida, argv[1]);
	
	std::string s = std::string( saida );
	if( s.substr(s.size() - 4, 4) == "tiff" ) {
		saida[strlen(saida)-5] = '\0';
	} else {
		saida[strlen(saida)-4] = '\0';
	}
	
	//saida[strlen(saida)-4] = '\0';
	string saida2(saida);
	
	string saida1 = saida2;
	string saida3 = saida2;
	string saida4 = saida3;
	
	saida1 += ".dogForHDR.distribution.txt";
	saida2 += ".dogForHDR1.txt";
	saida3 += ".dogForHDR2.txt";
	saida4 += ".dogForHDR3.txt";
	
	out0 = fopen(saida1.c_str(), "w+");
	out1 = fopen(saida2.c_str(), "w+");
	out2 = fopen(saida3.c_str(), "w+");
	out3 = fopen(saida4.c_str(), "w+");
	
	read(argv[1], argv[2]);
	
	//Inicializando octaves
	initOctaves();
	
	//Calculando DoG
	calcDoG();
	
	cout<<"dog\n";
	
	//Fazendo NonMaximaSupression na imagem DoG
	nonMaximaSupression();
	
	cout<<"nomMax\n";
	
	//Limiar na imagem de Response
	threshold();
	
	cout<<"threshold\n";
		
	//Limiar p/ edges response
	edgeThreshold(); 

	/*
	//Armengue para ver imagem hdr	---------------------------------------------------------
	for(int y = 0; y < input.rows; y++){
		for(int x = 0; x < input.cols; x++){
			input.at<Vec3f>(y, x)[0] *= 300;
			input.at<Vec3f>(y, x)[1] *= 300;
			input.at<Vec3f>(y, x)[2] *= 300;	
		}
	}
	*/
	
	//Salvando quantidade de Keypoints e para cada KP as coordenadas (x, y) e o response
	//saveKeypoints();
	saveKeypoints2ROIs();
	
	showKeyPoints();
	
	//Salvando imagens com Keypoints
	int len = strlen(saida);
	saida[len] = 'R';saida[len+1] = '.';saida[len+2] = 'j';saida[len+3] = 'p';saida[len+4] = 'g';saida[len+5] = '\0';
	imwrite(saida, input);
	
	cout<<"done!\n";
	
	//Mostrando imagem com Keypoints
	//imshow("Result", input);
	//waitKey(0);
	
	return 0;
}
