# Security policy

## Báo cáo lỗ hổng

Không mở issue công khai nếu lỗi có thể làm lộ token, dữ liệu hội thoại, tài
liệu nội bộ hoặc cho phép vượt xác thực. Hãy dùng **Security → Advisories → New
draft security advisory** trong repository GitHub và cung cấp:

- phiên bản hoặc commit bị ảnh hưởng;
- cách tái hiện tối thiểu, không kèm secret hay dữ liệu thật;
- tác động dự kiến và đề xuất khắc phục nếu có.

Maintainer sẽ xác nhận báo cáo, chuẩn bị bản vá riêng và công bố sau khi người
dùng có thời gian cập nhật.

## Phạm vi hỗ trợ

Nhánh `main` và release gần nhất được nhận bản vá. Dependency, model server và
dịch vụ bên ngoài có chính sách bảo mật riêng. Không commit `.env`, token,
database dump, log hội thoại, tài liệu riêng hoặc model weights vào repository.
