import requests
import time
import concurrent.futures
import numpy as np
import sys
import random
import json

class ApiBenchmark:
    
    def __init__(self, endpoint, workers):
        self.endpoint_url = endpoint
        self.max_workers = workers
        self.session = requests.Session()

    def _prepare_payloads(self, count):
        base_payloads = [
            # --- English (High-Speed Path) ---
            ("This is a perfectly clean and acceptable sentence.", False),
            ("You are a complete asshole and I hope you fail.", True),
            ("This is fucking brilliant, I can't believe it.", True),
            ("I absolutely love this, it's fantastic!", False),

            # --- German (Latin Script, Multilingual Path) ---
            ("Ich wünsche Ihnen einen schönen Tag.", False),
            ("Alter, du bist scheiße.", True),

            # --- Bengali (Bengali Script, Multilingual Path) ---
            ("আমি তোমাকে ভালোবাসি", False),
            ("তোমাকে চুদো", True),

            # --- Spanish (Latin Script, Multilingual Path) ---
            ("Eres un pendejo.", True),
            ("¡Qué tengas un buen día!", False),

            # --- Russian (Cyrillic Script, Multilingual Path) ---
            ("Ты просто сука.", True),
            ("Я желаю вам всего наилучшего.", False),

            # --- French (Latin Script, Multilingual Path) ---
            ("Tu es un connard.", True),
            ("Je vous souhaite une excellente journée.", False),

            # --- Japanese (CJK Script, Multilingual Path) ---
            ("この野郎！", True),
            ("今日はいい天気ですね。", False),
            
            # --- Cache-Test Duplicates ---
            ("Alter, du bist scheiße.", True),
            ("You are a complete asshole and I hope you fail.", True),
        ]
        return [random.choice(base_payloads) for _ in range(count)]

    def _execute_request(self, payload_tuple):
        text_payload, is_toxic = payload_tuple
        start_ts = time.perf_counter()
        validation_status = 'ERROR'
        try:
            with self.session.post(self.endpoint_url, data={'text': text_payload}, timeout=15) as response:
                response.raise_for_status()
                latency = (time.perf_counter() - start_ts) * 1000
                
                api_result = response.json()
                detected = len(api_result.get('detected_words', [])) > 0

                if is_toxic and detected:
                    validation_status = 'CORRECT'
                elif is_toxic and not detected:
                    validation_status = 'MISSED'
                elif not is_toxic and not detected:
                    validation_status = 'CORRECT'
                elif not is_toxic and detected:
                    validation_status = 'FALSE_POSITIVE'
                
                return latency, True, validation_status
        except (requests.RequestException, json.JSONDecodeError):
            return None, False, validation_status

    def _generate_report(self, title, latencies, duration, total_requests, validation_counts):
        successful_requests = len(latencies)
        failed_requests = total_requests - successful_requests
        
        report_lines = [f"\n--- {title} ---", "_"*40]
        
        if not successful_requests:
            report_lines.append("No successful requests to report.")
            print("\n".join(report_lines))
            return

        lat_np = np.array(latencies)
        rps = successful_requests / duration if duration > 0 else 0
        total_validated = validation_counts['CORRECT'] + validation_counts['MISSED'] + validation_counts['FALSE_POSITIVE']
        accuracy = (validation_counts['CORRECT'] / total_validated * 100) if total_validated > 0 else 0

        report_lines.extend([
            f"Total Duration:      {duration:.2f} s",
            f"Requests Sent:       {total_requests}",
            f"Successful:          {successful_requests}",
            f"Failed:              {failed_requests}",
            f"Requests Per Second: {rps:.2f} RPS",
            "_"*40,
            "Latency Statistics (ms):",
            f"  Average:           {np.mean(lat_np):.2f}",
            f"  Median (p50):      {np.median(lat_np):.2f}",
            f"  p95:               {np.percentile(lat_np, 95):.2f}",
            f"  p99:               {np.percentile(lat_np, 99):.2f}",
            f"  Min:               {np.min(lat_np):.2f}",
            f"  Max:               {np.max(lat_np):.2f}",
            "_"*40,
            "Accuracy Statistics:",
            f"  Correct Detections:  {validation_counts['CORRECT']}",
            f"  Missed Detections:   {validation_counts['MISSED']}",
            f"  False Positives:     {validation_counts['FALSE_POSITIVE']}",
            f"  Accuracy:            {accuracy:.2f}%",
            "_"*40
        ])
        print("\n".join(report_lines))

    def run_benchmark_suite(self, request_count):
        payloads = self._prepare_payloads(request_count)
        
        for bench_type in ["Sequential", "Concurrent"]:
            latencies = []
            validation_counts = {'CORRECT': 0, 'MISSED': 0, 'FALSE_POSITIVE': 0, 'ERROR': 0}
            title = f"{bench_type} Multilingual Benchmark"
            if bench_type == "Concurrent":
                title += f" (Workers: {self.max_workers})"

            start_time = time.perf_counter()
            
            if bench_type == "Sequential":
                for payload in payloads:
                    latency, success, status = self._execute_request(payload)
                    if success:
                        latencies.append(latency)
                        validation_counts[status] += 1
                    else:
                        validation_counts['ERROR'] += 1
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_payload = {executor.submit(self._execute_request, p): p for p in payloads}
                    for future in concurrent.futures.as_completed(future_to_payload):
                        latency, success, status = future.result()
                        if success:
                            latencies.append(latency)
                            validation_counts[status] += 1
                        else:
                            validation_counts['ERROR'] += 1

            end_time = time.perf_counter()
            self._generate_report(title, latencies, end_time - start_time, request_count, validation_counts)
    
    def warmup_server(self, count=10):
        print("Warming up server with multilingual payloads...")
        payloads = self._prepare_payloads(count)
        for p in payloads: self._execute_request(p)
        print("Warm-up complete.")


def main():
    api_endpoint = "http://127.0.0.1:8000/censor"
    
    try:
        num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    except (ValueError, IndexError):
        print("Usage: python benchmark_multilingual.py <num_requests> <concurrency>")
        sys.exit(1)

    print(f"Target: {api_endpoint} | Requests: {num_requests} | Concurrency: {concurrency}")
    
    benchmark_runner = ApiBenchmark(endpoint=api_endpoint, workers=concurrency)
    
    benchmark_runner.warmup_server()
    benchmark_runner.run_benchmark_suite(request_count=num_requests)


if __name__ == "__main__":
    main()