# Huong dan thu nghiem

## Muc tieu
- Danh gia do ben cua mo hinh phan loai OPIXray truoc FGSM va PGD.

## Quy trinh
1. Train mo hinh clean.
2. Danh gia clean accuracy.
3. Chay FGSM voi nhieu muc eps.
4. Chay PGD voi nhieu muc eps.
5. So sanh ket qua va rut ra nhan xet.

## Mau bang ket qua
| attack | eps | acc | attack_success_rate | avg_l2 | avg_linf | avg_psnr | avg_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | ... | 0.0 | 0.0 | 0.0 | inf | 1.0 |
| fgsm | 0.1 | ... | ... | ... | ... | ... | ... |
| pgd | 0.1 | ... | ... | ... | ... | ... | ... |

## Cau hoi thao luan
- Tai sao PGD thuong manh hon FGSM?
- Do chinh xac giam theo eps co tuyen tinh khong?
- PSNR va SSIM giam nhu the nao khi eps tang?
- Neu huan luyen doi khang thi ket qua thay doi nhu the nao?
