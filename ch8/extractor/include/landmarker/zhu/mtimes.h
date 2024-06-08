//#pragma once
//#include "stdafx.h"
//class _mtimes
//{
//public:
//	LARGE_INTEGER Freq;//64位有符号整数值. // 获取时钟周期  “1次/秒”，记做Hz（赫兹）。1Hz就是每秒一次
//	LARGE_INTEGER start;  // 获取时钟计数
//	LARGE_INTEGER end;
//	vector<double> step_time;
//public:
//	_mtimes(){QueryPerformanceFrequency(&Freq);}
//	double calcFreq(){return (double)Freq.QuadPart/(double)((end.QuadPart-start.QuadPart));}
//	double calcTime()
//	{
//		/*此处*1000，以毫秒为单位；*1000000 以微秒为单位*/
//		/*由于执行时间极短（可能是几微秒），所以采用微秒为单位*/
//		/*  1s=10^3ms(毫秒)=10^6μs(微秒)=10^9ns(纳秒)  */
//		//printf("%d\n",(end.QuadPart-start.QuadPart)*1000000/Freq.QuadPart);
//		return (double)((end.QuadPart-start.QuadPart)*1000/Freq.QuadPart);
//	}
//	double aveFreq()
//	{
//		double aveFrq=0;
//		for (int i=0;i<step_time.size();i++)
//			aveFrq+=step_time[i];
//		return (double)1000000*(double)step_time.size()/aveFrq;
//	}
//};
