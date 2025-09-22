import asyncio
import concurrent.futures
import contextlib
import threading
from concurrent.futures import Future
from typing import Any, Coroutine


class AsyncBackgroundExecutor:
    """可重用的异步任务执行器, 将协程提交到运行于后台线程的事件循环中执行"""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop
        self._pending_futures: set[Future] = set()
        self._lock = threading.Lock()
        self._running = False
        self._start_background_thread()

    def submit[T](self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """提交一个协程到后台线程执行, 返回一个 Future 对象. 此方法线程安全且非阻塞."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._remove_future)
        with self._lock:
            self._pending_futures.add(future)
        return future

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """submit 的阻塞版本, 等待协程执行完毕并返回结果. 此方法线程安全."""
        return self.submit(coro).result()

    def wait_all(self, timeout=None):
        """等待所有未完成任务执行完毕, 任务异常将作为结果返回. 此方法线程安全."""
        with self._lock:
            current_futures = list(self._pending_futures)
            if not current_futures:
                return []
        try:
            done, not_done = concurrent.futures.wait(
                current_futures,
                timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            if not_done:
                raise TimeoutError(f"{len(not_done)} tasks not completed within timeout")
            results = []
            for future in done:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(e)
            return results
        except Exception:
            for future in current_futures:
                if not future.done():
                    future.cancel()
            raise

    def cancel(self):
        """取消所有任务"""
        with self._lock:
            if not self._running:
                return
            self._running = False
            _pending_futures = list(self._pending_futures)

        # 取消所有待处理的任务
        for future in _pending_futures:
            if not future.done():
                future.cancel()  # 此处会运行 callback _remove_future
        self._running = True  # 此方法不关闭后台线程和事件循环, 仅取消任务

    def cancel_async(self):
        """取消所有任务. cancel 的非阻塞版本

        提交一个新的任务到事件循环中去取消其他任务
        """
        with self._lock:
            if not self._running:
                return
            _pending_futures = list(self._pending_futures)
            if not _pending_futures:
                return

        # 创建一个异步任务来取消所有待处理任务
        async def _cancel_all():
            for future in _pending_futures:
                if not future.done():
                    future.cancel()
            return True

        # 提交取消任务到事件循环
        return self.submit(_cancel_all())

    def _run_event_loop(self):
        """运行事件循环的线程函数"""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_started.set()
            self._loop.run_forever()
        except Exception:
            # 如果启动失败，设置事件以避免主线程永远等待
            self._loop_started.set()
            # 不重新抛出异常，让线程正常结束
        finally:
            # 清理资源
            if self._loop and not self._loop.is_closed():
                with contextlib.suppress(Exception):
                    self._loop.close()
            # 重置loop引用
            self._loop = None  # type: ignore

    def _remove_future(self, future):
        """自动移除已完成的任务"""
        with self._lock:
            self._pending_futures.discard(future)

    def _start_background_thread(self):
        self._loop_started = threading.Event()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True, name="AsyncBackgroundThread")
        self._thread.start()
        self._running = True
        # 等待事件循环启动
        if not self._loop_started.wait(timeout=10.0):
            raise RuntimeError("Failed to start background event loop within 10 seconds")

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.cancel()
            self._loop.stop()
            self._thread.join(timeout=5.0)
        except Exception:
            pass  # 忽略析构时的异常
