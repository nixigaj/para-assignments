#include <iostream>
#include <thread>
#include <mutex>

std::mutex mutex;

void loop(const int n) {
	mutex.lock();
	std::cout << "Task " << n << " is running." << std::endl;
	mutex.unlock();

	mutex.lock();
	std::cout << "Task " << n << " is terminating." << std::endl;
	mutex.unlock();
}

int main() {
	std::thread t1(loop, 1);
	std::thread t2(loop, 2);
	std::thread t3(loop, 3);
	std::thread t4(loop, 4);
	std::thread t5(loop, 5);
	std::thread t6(loop, 6);
	std::thread t7(loop, 7);
	std::thread t8(loop, 8);

	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();
	t8.join();

	return 0;
}
