#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mutex;

long x = 0;
bool run = true;

void inc() {
	bool r = true;
	while (r)
	{
		mutex.lock();
		++x;
		r = run;
		mutex.unlock();
	}
}

void dec() {
	bool r = true;
	while (r)
	{
		mutex.lock();
		--x;
		r = run;
		mutex.unlock();
	}
}

void print() {
	bool r = true;
	while (r)
	{
		mutex.lock();
		std::cout << x << std::endl;
		r = run;
		mutex.unlock();
	}
}

int main() {
	std::thread t1(inc);
	std::thread t2(dec);
	std::thread t3(print);

	std::this_thread::sleep_for(std::chrono::seconds(1));

	mutex.lock();
	run = false;
	mutex.unlock();

	t1.join();
	t2.join();
	t3.join();

	return 0;
}
