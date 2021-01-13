#include "../include/detectors/dog.h"

void dogInitScales(cv::Mat img, cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], int mgauss)
{	
	cv::Mat img_aux;
	float k[] = {0.707107, 1.414214, 2.828428, 5.656856};
	
	img.convertTo(img_aux, CV_32FC1);
	
	for(int i = 0; i < DOG_SCL_ROWS; i++)
	{
		float ko = k[i];
		for(int j = 0; j < DOG_SCL_COLS; j++)
		{			
			GaussianBlur(img_aux, scales[i][j], cv::Size(mgauss, mgauss), ko, ko, cv::BORDER_DEFAULT);
			ko = ko*1.414214;
		}
		cv::resize(img_aux, img_aux, cv::Size(img_aux.cols/2, img_aux.rows/2));
	}
}

void dogCalc(cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1])
{
	for(int i = 0; i < DOG_SCL_ROWS; i++)
		for(int j = 0; j < DOG_SCL_COLS - 1; j++)
		{
			dog[i][j] = cv::Mat::zeros(scales[i][j].size(), CV_32FC1);
			cv::subtract(scales[i][j], scales[i][j + 1], dog[i][j]);
		}
}

void dogMaxSup(cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], cv::Mat roi[], std::vector<KeyPoints> &kp, int maxsup_size)
{
	int maxsup_rad = maxsup_size/2;

	for(int s = 0; s < DOG_SCL_ROWS; s++)
	{
		for(int l = 1; l < DOG_SCL_COLS - 1; l++)
		{
			cv::Mat middle = dog[s][l];
			cv::Mat down = dog[s][l - 1];
			cv::Mat up = dog[s][l + 1];
			cv::Mat dog_aux = cv::Mat::zeros(middle.size(), CV_32FC1);
			
			for(int y = maxsup_rad; y < middle.rows - maxsup_rad; y++)
			{
				for(int x = maxsup_rad; x < middle.cols - maxsup_rad; x++)
				{
					if(roi[0].at<uchar>(y*pow(2, s), x*pow(2, s)) == 0)
						continue;
					
					float curr_px = middle.at<float>(y, x);
					bool is_smaller = true;
					bool is_bigger = true;
					
					for(int i = y - maxsup_rad; i <= y + maxsup_rad; i++)
					{
						for(int j = x - maxsup_rad; j <= x + maxsup_rad; j++)
						{
							if(!((curr_px < middle.at<float>(i, j) || (y == i && x == j)) && 
								 (curr_px < down.at<float>(i, j)) &&
								 (curr_px < up.at<float>(i, j))))
							{
								is_smaller = false;
								break;
							}
						}
						if(!is_smaller)
							break;
					}
					for(int i = y - maxsup_rad; i <= y + maxsup_rad; i++)
					{
						for(int j = x - maxsup_rad; j <= x + maxsup_rad; j++)
						{
							if(!((curr_px > middle.at<float>(i, j) || (y == i && x == j)) && 
								 (curr_px > down.at<float>(i, j)) && 
								 (curr_px > up.at<float>(i, j))))
							{
								is_bigger = false;
								break;
							}
						}
						if(!is_bigger)
							break;
					}
					if(is_smaller || is_bigger)
						kp.push_back({y, x, curr_px, s, l});
						
				}
			}
		}
	}
}

void contrastThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th)
{
	std::vector<KeyPoints> kp_aux;
	
	for(int i = 0; i < kp.size(); i++)
	{
		if(kp[i].resp >= contrast_th)
			kp_aux.push_back(kp[i]);	
	}
	kp.clear();
	kp = kp_aux;
}

void edgeThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float curv_th)
{
	std::vector<KeyPoints> kp_aux;
	curv_th = (curv_th + 1)*(curv_th + 1)/curv_th;
	
	for(int i = 0; i < kp.size(); i++)
	{
		cv::Mat D = dog[kp[i].scale][kp[i].level];
		
		int y = kp[i].y;
		int x = kp[i].x;
		
		float dxx = D.at<float>(y - 1, x) + D.at<float>(y + 1, x) - 2.0*D.at<float>(y, x);
		float dyy = D.at<float>(y, x - 1) + D.at<float>(y, x + 1) - 2.0*D.at<float>(y, x);
		float dxy = 0.25*(D.at<float>(y - 1, x - 1) + D.at<float>(y + 1, x + 1) - D.at<float>(y + 1, x - 1) - D.at<float>(y - 1, x + 1));

		float trH = dxx*dyy;
		float detH = dxx*dyy - dxy*dxy;

		float curv_ratio = trH*trH/detH;
		
		if((detH > 0) && (curv_ratio > curv_th))
			kp_aux.push_back(kp[i]);
	}
	kp.clear();
	kp = kp_aux;
}

void dogThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th, float curv_th)
{
	contrastThreshold(kp, dog, contrast_th);
	edgeThreshold(kp, dog, curv_th);
}

void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoints> &kp, int mgauss, int maxsup_size, float contrast_th, float curv_th)
{
	cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS];
	cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1];

	dogInitScales(img, scales, mgauss);
	dogCalc(scales, dog);
	dogMaxSup(dog, roi, kp, maxsup_size);
	dogInterpExtrem(); // ainda vamos ver quais sao os argumentos
	dogThreshold(kp, dog, contrast_th, curv_th);
}

/*
  Interpolates a scale-space extremum's location and scale to subpixel
  accuracy to form an image feature.  Rejects features with low contrast.
  Based on Section 4 of Lowe's paper.  

  @param dog_pyr DoG scale space pyramid
  @param octv feature's octave of scale space
  @param intvl feature's within-octave interval
  @param r feature's image row
  @param c feature's image column
  @param intvls total intervals per octave
  @param contr_thr threshold on feature contrast

  @return Returns the feature resulting from interpolation of the given
		  parameters or NULL if the given location could not be 
		  interpolated or if contrast at the interpolated loation was 
		  too low.  If a feature is returned, its scale, orientation,
		  and descriptor are yet to be determined.
*/
static struct feature* interp_extremum( cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], int octv,
										int intvl, int r, int c, 
										int intvls, double contr_thr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double xi, xr, xc, contr;
  int i = 0;
  
  while( i < SIFT_MAX_INTERP_STEPS )
    {
      interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
      if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
	break;
      
      c += cvRound( xc );
      r += cvRound( xr );
      intvl += cvRound( xi );
      
      if( intvl < 1  ||
	  intvl > intvls  ||
	  c < SIFT_IMG_BORDER  ||
	  r < SIFT_IMG_BORDER  ||
	  c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER  ||
	  r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER )
	{
	  return NULL;
	}
      
      i++;
    }
  
  /* ensure convergence of interpolation */
  if( i >= SIFT_MAX_INTERP_STEPS )
    return NULL;
  
  contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
  if( ABS( contr ) < contr_thr / intvls )
    return NULL;

  feat = new_feature();
  ddata = feat_detection_data( feat );
  feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );
  feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
  ddata->r = r;
  ddata->c = c;
  ddata->octv = octv;
  ddata->intvl = intvl;
  ddata->subintvl = xi;

  return feat;
}



/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl interval being interpolated
  @param r row being interpolated
  @param c column being interpolated
  @param xi output as interpolated subpixel increment to interval
  @param xr output as interpolated subpixel increment to row
  @param xc output as interpolated subpixel increment to col
*/

static void interp_step( IplImage*** dog_pyr, int octv, int intvl, int r, 
						 int c, double* xi, double* xr, double* xc )
{
  CvMat* dD, * H, * H_inv, X;
  double x[3] = { 0 };
  
  dD = deriv_3D( dog_pyr, octv, intvl, r, c );
  H = hessian_3D( dog_pyr, octv, intvl, r, c );
  H_inv = cvCreateMat( 3, 3, CV_64FC1 );
  cvInvert( H, H_inv, CV_SVD );
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );
  
  cvReleaseMat( &dD );
  cvReleaseMat( &H );
  cvReleaseMat( &H_inv );

  *xi = x[2];
  *xr = x[1];
  *xc = x[0];
}
