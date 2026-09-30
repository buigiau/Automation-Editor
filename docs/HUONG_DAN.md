# Test Soda Pop và các project cùng cấu trúc

## Bản đã tạo sẵn

Bản chỉ thay sample nằm ở `output/samples-only-270926/`, dựng lại từ `D:\sodapop\270926`:

- Mở `audio_review.html` để nghe âm gốc và voice của từng track.
- `edit-plan.json` giữ nguyên điểm cắt, cấu trúc, timeline và ánh xạ 16 track. Chỉ đổi lựa chọn/sample WAV; `I` → `ai/aii`, `rô.wav` giảm từ 8 xuống 2 track.
- Sample 2 và 16 đã bỏ 0,94 giây im lặng đầu. WAV mới có tiếng từ đầu; chưa nạp/nghe bản phối mới trong Cubase.
- Không Apply hoặc import mixdown vào Premiere cho lần này. Trong GUI, đặt Output Folder là `output/samples-only-270926`, mở Cubase template đi kèm rồi bấm **Import Cubase**. Khởi động lại GUI sau cập nhật mã. Bản `output/repaired-270926` đã bị thay thế vì có dịch điểm cắt.
- Các mục bên dưới nói về bản dựng ngày 26/09, không phải bản sửa mới này.

- `output/edit-plan.json`: 14 cảnh chính + intro 0–1 giây; 16 lượt âm thanh từ cảnh 1 ở giây 1.
- `output/cubase_imports/20260926_194904_467063/soda pop_autoedit.cpr`: bản sao đã nạp 16 sample và kiểm chứng MIDI, tham số sampler, hiệu ứng.
- `output/sampler_mapping.csv`: file âm thanh ứng với từng cảnh/track.
- `output/voice_catalog.csv`: nhãn phiên âm, loại tiếng và khoảng có âm của 127 file Voice.

1. Trong UXP Developer Tool, **Unload → Load** lại plugin tại `plugins/premiere-uxp/manifest.json`.
2. Mở Premiere project Soda Pop mẫu đã gửi ngày 26/09, mở panel AutoEdit, **Load plan** → chọn `output/edit-plan.json`.
3. Bấm **Fill nested clips**. Kết quả gồm 14 nest và một đoạn intro riêng, không chèn âm thanh nguồn. Các nhịp cắt/repeat của timeline vẫn giữ nguyên. Tự Save As trong Premiere khi đã kiểm tra.
4. Mở bản sao Cubase ở trên để nghe phần voice. Không cần nạp lại 16 file bằng tay cho bản này.
5. Xuất phần voice từ **0 đến 65,8 giây** thành `output/cubase_mixdown.wav`, giữ đoạn im lặng đầu. Không xuất trùng beat nếu Premiere A1 đã có beat.
6. Trong panel Premiere, bấm **Import mixdown WAV**. Audio được đặt ở giây 0, có sẵn khoảng im lặng trước cảnh 1 và được giới hạn theo timeline. Không tự kéo file này tới giây 1 lần nữa.

Khớp khẩu hình là ước lượng theo hình môi và nhịp âm; chưa xác nhận đọc đúng từng âm vị. Bản hiện tại đánh dấu cảnh **1, 3, 5, 11** cần nghe/xem lại; cảnh 3 lặp ở track 15 cũng dùng lại mẫu đó. Các lựa chọn dự phòng nằm trong `audio_match.alternatives` của plan. File preview Python không chứa MIDI/pitch/hiệu ứng Cubase.

## Chạy project/video mới

1. Mở tool:

   ```powershell
   .\.venv\Scripts\python.exe -m autoedit gui
   ```

2. Chọn Premiere project, Cubase project đi kèm và video nguồn. Audio Folder mặc định là `D:\Editor\Voice`. Chọn output; giữ cùng thư mục để tận dụng cache. Đặt Source gap, ví dụ **10**.
3. Bấm **Run**, đợi log báo `Run completed`. Chỉ những cảnh đã chọn mới được soi môi kỹ ở 20 mẫu/giây; tool không quét dày toàn bộ video dài.
4. Mở project đi kèm trong **Cubase 13 giao diện tiếng Anh, Windows scale 100%**. Tự chọn đủ các Sampler Track đánh số trước khi bấm Import Cubase. Bước Ctrl+F / Shift+Down tự chọn track đã bỏ theo yêu cầu.
5. Bấm **Import Cubase** trong tool. Để yên chuột/phím trong lúc nhập. Tool xuất cấu hình, nạp sample theo tên track vào project đang mở, rồi export lại để kiểm chứng. Tool không Save As hoặc tự lưu project; kiểm tra trong Cubase và tự Save nếu muốn giữ thay đổi. Sau khi xác minh thành công, XML và ảnh tạm được dọn, báo cáo xác minh vẫn được giữ. Không cần tự export XML trước.
6. Load plan mới trong Premiere → **Fill nested clips**. Nghe/xem lại, xuất voice từ Cubase rồi **Import mixdown WAV** như trên.

Nếu intro trước cảnh 1 đang trống, tool tự cắt thêm đúng một cảnh đủ độ dài. Nếu đã có nội dung/đồ họa ở đó, tool giữ nguyên và báo trong log. Số Sampler Track luôn tính từ các lượt cảnh chính, không tính intro.

Với hoạt hình, chọn `animation` và dùng thư mục Voice chung. Tool tìm từ nguồn trong toàn bộ thư viện trước, rồi mới dùng nhóm NEUTRAL khi cần dự phòng.

Nếu import báo thiếu/sai track, tự chọn lại đủ Sampler Track rồi bấm Import Cubase; không cần Run lại phần phân tích video. CLI `autoedit cubase-import đường-dẫn/edit-plan.json` cũng luôn dùng lựa chọn bằng tay. Đóng/mở lại GUI sau cập nhật để bỏ luồng tự chọn cũ.

Không đưa khoảng im lặng để căn đầu từ vào WAV sampler: nốt MIDI ngắn có thể kết thúc trước lúc có tiếng. Preview giữ độ lệch của từ nguồn, còn sampler phát ngay phần voice theo MIDI của template. Mở `audio_review.html` để nghe so sánh; sau đó vẫn cần nghe bản phối thực tế trong Cubase.
