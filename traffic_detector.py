from ultralytics import YOLO
import urllib.request
import os

def run_traffic_detection():
    print("🚗 در حال راه‌اندازی سیستم بینایی ماشین (YOLOv8)...")
    
    # 1. دانلود مدل پیش‌آموزش‌دیده (نسخه سبک برای اجرای سریع)
    # اگر فایل وجود نداشته باشد، خودش دانلود میکند
    model = YOLO('yolov8n.pt') 
    
    # 2. دانلود یک عکس نمونه از ترافیک سنگین برای تست
    image_url = "https://ultralytics.com/images/bus.jpg" # عکس استاندارد ترافیک
    image_path = "test_traffic.jpg"
    
    print("📥 در حال دانلود تصویر تست ترافیک...")
    urllib.request.urlretrieve(image_url, image_path)
    
    # 3. اجرای هوش مصنوعی روی عکس
    print("🧠 در حال پردازش تصویر و تشخیص خودروها...")
    results = model(image_path)
    
    # 4. ذخیره نتیجه
    for result in results:
        # ساخت یک نام جدید برای عکس خروجی
        output_filename = "output_detected_traffic.jpg"
        result.save(filename=output_filename)  # ذخیره عکس با باکس‌های رنگی
        
        # چاپ آمار برای ترمینال
        detected_objects = len(result.boxes)
        print(f"\n✅ پردازش تمام شد! تعداد {detected_objects} شیء در تصویر تشخیص داده شد.")
        print(f"🖼️ تصویر خروجی در فایل '{output_filename}' ذخیره شد.")

if __name__ == "__main__":
    run_traffic_detection()