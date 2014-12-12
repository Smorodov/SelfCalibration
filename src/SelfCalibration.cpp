#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;
// ----------------------------------------------------------------
// Забодало писать по сто раз одинаковые вещи, соорудил макросы :)
// ----------------------------------------------------------------
#define COUT_VAR(x) cout << #x"=" << x << endl;
#define SHOW_IMG(x) namedWindow(#x);imshow(#x,x);waitKey(20);
// ----------------------------------------------------------------
// Логарифм по основанию 2
// ----------------------------------------------------------------
double log2( double n )  
{   
	return log( n ) / log( 2.0 );  
}
//----------------------------------------------------------
// Frobenius norm
//----------------------------------------------------------
double FNorm(Mat& M)
{
return (sqrt(sum((M*M.t()).diag())[0]));
}
// ----------------------------------------------------------------
// Применение изображению img преобразования Tau
// ----------------------------------------------------------------
void transformImg(Mat& img,Mat& Tau,Mat& dst)
{	
	double cx=Tau.at<double>(0);
	double cy=Tau.at<double>(1);
	double fx=Tau.at<double>(2);
	double fy=Tau.at<double>(3);

	Mat cameraMatrix=(Mat_<double>(3, 3) << 
		fx, 0, cx,
		0,  fy, cy,
		0,  0, 1);

	Mat distCoeffs(4,1,CV_64FC1);

	distCoeffs.at<double>(0)=Tau.at<double>(4);
	distCoeffs.at<double>(1)=Tau.at<double>(5);
	distCoeffs.at<double>(2)=Tau.at<double>(6);
	distCoeffs.at<double>(3)=Tau.at<double>(7);

	cv::undistort(img,dst,cameraMatrix,distCoeffs);
}
//----------------------------------------------------------
// Функция стоимости (которую мы минимизируем)
//----------------------------------------------------------
double getF(Mat& img)
{
	Mat tmp=img/norm(img);
	Mat A, w, u, vt;
	SVD::compute(img, w, u, vt,SVD::NO_UV);
	return sum(w)[0];
}
//----------------------------------------------------------
// Якобиан изображения по матрице преобразования
// Производные изображения по элементам матрицы преобразования
// располагаются по столбцам: 1 столбец - одна производная.
// Tau - матрица преобразования.
//----------------------------------------------------------
void getJacobian(Mat& img,Mat& Tau,Mat& J_vec)
{
	// Количество степеней свободы преобразования
	int p=8;
	// Зададим малую величину, для вычисления производных численным методом.
	// Разные параметры имеют разную чувствительность, поэтому для каждого сделано персонально.
	double epsilon[8]={1, // x
					   1, // y
					   1, // fx
					   1, // fy
					   2e-1, // k1
					   2e-1, // k2
					   5e-1, // k3
					   5e-1  // k4
	};
	Mat dst;
	Mat Ip,Im,Tau_p,Tau_m,diff;
	// Размеры области интереса
	int n=img.cols;
	int m=img.rows;
	// Матрица результата
	J_vec=Mat::zeros(n*m,p,CV_64FC1);
	// ---------------------------------------------------------------------
	// Вычисляем 8 частных производных по элементам матрицы преобразования.
	// Девятый элемент равен 1
	// ---------------------------------------------------------------------
	for(int i=0;i<p;i++)
	{
		Tau_p=Tau.clone();
		Tau_m=Tau.clone();
		Tau_p.at<double>(i)+=epsilon[i];
		Tau_m.at<double>(i)-=epsilon[i];
		transformImg(img,Tau_p,Ip);
		transformImg(img,Tau_m,Im);
		// Разница изображений
		diff=Ip-Im;
		// Уменьшим шум.
		cv::GaussianBlur(diff,diff,Size(3,3),2);
		
		diff/=2*epsilon[i];
		diff=diff.reshape(1,m*n);		
		
		diff.copyTo(J_vec.col(i));
	}
}

//--------------------------------
// 
//--------------------------------
double sign(double X)
{
	double res=0;
	if(X>0){res=1;}
	if(X<0){res=-1;}
	return res;
}
//--------------------------------
// Ослабление коэффициентов
//--------------------------------
double S_mu(double d,double mu)
{
	double res;
	res=sign(d)*MAX(fabs(d)-mu,0);
	return res;
}
//----------------------------------------------------------
// Функция подавления слабых коэффициентов для матриц
//----------------------------------------------------------
Mat Smu(Mat& x,double mu)
{
	Mat res(x.size(),CV_64FC1);
	for(int i=0;i<x.rows;i++)
	{
		for(int j=0;j<x.cols;j++)
		{
			res.at<double>(i,j)=S_mu(x.at<double>(i,j),mu);
		}
	}
	return res;
}
//----------------------------------------------------------
// Главная функция (augmented Lagrange multiplier method)
//----------------------------------------------------------
Mat ALM(Mat& I,Mat& Tau_0,double Lambda)
{
	int maxcicles=100;
	int cicle_counter1=0;
	int cicle_counter2=0;
	bool converged1=0;
	bool converged2=0;
	Mat I_tau;
	Mat Tau=Tau_0.clone();
	Mat deltaTau_prev;
	Mat deltaTau;

	double F_prev=DBL_MAX;

	int n=I.cols;
	int m=I.rows;
	int p=8;

	double mu;
	double rho=1.5; // Каждый цикл порог увеличивается
	Mat J_vec;

	while(!converged1)
	{
		// Шаг 1 Получаем нормированное преобразованное изображение и якобиан
		transformImg(I,Tau,I_tau);
		getJacobian(I,Tau,J_vec);

		// Шаг 2 Решение линеаризованной задачи оптимизации
		Mat E=Mat::zeros(m, n,CV_64FC1);
		Mat A=Mat::zeros(m, n,CV_64FC1);

		// Начальное приближение порога ослабления
		mu=1.25/norm(I_tau);


		deltaTau=Mat::zeros(p, 1,CV_64FC1);
		deltaTau_prev=Mat::zeros(p, 1,CV_64FC1);

		Mat Y=I_tau.clone();
		Mat I0;

		cicle_counter2=0;
		converged2=0;

		Mat J_vec_inv=J_vec.inv(DECOMP_SVD);

		while(!converged2)
		{
			Mat t1=J_vec*deltaTau;
			Mat tmp;
			Mat U, Sigma,V;
			SVD::compute(I_tau+t1.reshape(1,m)-E+Y/mu,Sigma,U,V);
			Sigma=Smu(Sigma,1.0/mu);// Урезали собственные значения

			// собираем матрицу Sigma (по-диагонали собственные числа, остальное нули)
			// так как функция вернула нам их в виде вектора, а нам нужна диагональная матрица
			Mat W=Mat::zeros(Sigma.rows,Sigma.rows,CV_64FC1);
			Sigma.copyTo(W.diag());

			I0=U*W*V;// Собрали обратно

			tmp=I_tau+t1.reshape(1,m)-I0+Y/mu;
			E=Smu(tmp,Lambda/mu);

			tmp=(-I_tau+I0+E-Y/mu);
			deltaTau=J_vec_inv*tmp.reshape(1,m*n);

			tmp=J_vec*deltaTau;			
			Y+=mu*(I_tau+tmp.reshape(1,m)-I0-E);

			mu*=rho;

			// Ограничение по количеству внутренних циклов
			cicle_counter2++;
			if(cicle_counter2>maxcicles){break;}

			// Проверка, сошлось или нет.
			double m,M;
			cv::minMaxLoc(abs(deltaTau_prev-deltaTau),&m,&M);	
			if(cicle_counter2>1 && M<1e-3)
			{
				converged2=true;
			}
			// Это чтобы фиксировать динамику процесса
			deltaTau_prev=deltaTau.clone();
		}
		// Шаг 3 Уточняем преобразование
		for(int i=0;i<p;i++)
		{
			Tau.at<double>(i)+=deltaTau.at<double>(i);
		}

		Mat dst;
		transformImg(I,Tau,dst);

		double F=getF(dst);
		double perf=F_prev-F;
		F_prev=F;
		// Целевая функция
		COUT_VAR(F);

		// ----------------------------------
		// Изображение
		dst.convertTo(dst,CV_8UC1,255);
		imshow("Undistorted",dst);
		cvWaitKey(10);
		// ----------------------------------

		// Проверка, сошлось или нет.
		if(perf<=0){converged1=true;}

		// Ограничение по количеству внешних циклов
		cicle_counter1++;
		if(cicle_counter1>maxcicles){break;}

	}
	return Tau;
}
//----------------------------------------------------------
// Функция получения координат области (2 угловые точки)
// Выбрать 2 точки и нажать кнопку
//----------------------------------------------------------
vector<Point2f> pt;
void pp_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if(event == CV_EVENT_LBUTTONDOWN)
	{
		pt.push_back(Point2f(x,y));
	}
}
//----------------------------------------------------------
// Рисуем область по 4 точкам
//----------------------------------------------------------
void DrawRegion(Mat& img,vector<Point2f>& region,Scalar color)
{
	for(int i = 0; i < 4; i++ )
	{
		line(img, region[i], region[(i+1)%4], color, 1, CV_AA);
	}
}

//----------------------------------------------------------
// Точка входа
//----------------------------------------------------------
int main(int argc, char* argv[])
{
	// Матрица изображения
	Mat img_c,img;
	//---------------------------------------------
	// Инициализация
	//---------------------------------------------	
	namedWindow("Исходное изображение");
	namedWindow("Undistorted");

	// Грузим изображение
	img_c=imread("2.jpg",1);
	resize(img_c,img_c,Size(200,200));
	// Ввод точек области
	imshow("Исходное изображение", img_c);
	waitKey(10);

	//setMouseCallback("Исходное изображение",pp_MouseCallback,0);
	//waitKey(0);

	// Переводим в формат серого изображения с плавающей точкой
	cv::cvtColor(img_c,img,cv::COLOR_BGR2GRAY);
	// В интервал 0-1
	img.convertTo(img,CV_64FC1,1.0/255.0);

	// Рисуем рамку, введенную пользователем
	//DrawRegion(img_c, region,Scalar(0, 255, 0));

	imshow("Исходное изображение", img_c);

	// Ищем начальное приближение (можно не искать, а просто начать с нулевых значений)
	// Но так, наверное лучше.

	double theta_opt=0;
	double t_opt=0;
	Mat dst;

	Mat Tau_0(8,1,CV_64FC1);
	Tau_0.at<double>(0)=img.cols/2;
	Tau_0.at<double>(1)=img.rows/2;
	Tau_0.at<double>(2)=img.cols;
	Tau_0.at<double>(3)=img.rows;
	Tau_0.at<double>(4)=0;
	Tau_0.at<double>(5)=0;
	Tau_0.at<double>(6)=0;
	Tau_0.at<double>(7)=0;

	Mat Tau=ALM(img,Tau_0,0.5);

	// Результат коррекции: cx,cy,fx,fy,k1,k2,p1,p2
	COUT_VAR(Tau);

	imshow("Исходное изображение", img_c);
	cvWaitKey(0);
	destroyAllWindows();
	return 0;
}
