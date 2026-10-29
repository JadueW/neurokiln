import QRCode from 'qrcode'
import i18n from '../i18n'
import { tr } from '../i18n/catalog'

const WIDTH = 1200
const HEIGHT = 630
const FONT = '"PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif'

function label(zh, en) {
  return i18n.global.locale.value === 'en' ? en : zh
}

function demoUrl() {
  return import.meta.env.VITE_DEMO_URL || `${window.location.origin}/demo/index.html`
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2)
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + width, y, x + width, y + height, r)
  ctx.arcTo(x + width, y + height, x, y + height, r)
  ctx.arcTo(x, y + height, x, y, r)
  ctx.arcTo(x, y, x + width, y, r)
  ctx.closePath()
}

function clipText(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text
  let value = text
  while (value.length > 1 && ctx.measureText(`${value}…`).width > maxWidth) {
    value = value.slice(0, -1)
  }
  return `${value}…`
}

function formatNumber(value, digits = 3) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : '—'
}

function bestRow(detail) {
  const rows = [...(detail.rows || [])]
  rows.sort((a, b) => Number(b.balanced_accuracy_mean || 0) - Number(a.balanced_accuracy_mean || 0))
  return rows[0] || null
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = reject
    image.src = source
  })
}

export async function buildShareCard(detail) {
  const canvas = document.createElement('canvas')
  canvas.width = WIDTH
  canvas.height = HEIGHT
  const ctx = canvas.getContext('2d')
  const best = bestRow(detail)
  const rows = detail.rows || []
  const folds = Math.max(...rows.map((row) => row.n_splits || 0), 0)

  const background = ctx.createLinearGradient(0, 0, WIDTH, HEIGHT)
  background.addColorStop(0, '#0a1424')
  background.addColorStop(0.55, '#102a45')
  background.addColorStop(1, '#123c5c')
  ctx.fillStyle = background
  ctx.fillRect(0, 0, WIDTH, HEIGHT)

  ctx.strokeStyle = 'rgba(126, 186, 255, 0.08)'
  ctx.lineWidth = 1
  for (let x = -HEIGHT; x < WIDTH; x += 46) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x + HEIGHT, HEIGHT)
    ctx.stroke()
  }
  const glow = ctx.createRadialGradient(980, 90, 10, 980, 90, 420)
  glow.addColorStop(0, 'rgba(72, 160, 255, 0.32)')
  glow.addColorStop(1, 'rgba(72, 160, 255, 0)')
  ctx.fillStyle = glow
  ctx.fillRect(0, 0, WIDTH, HEIGHT)

  const mark = ctx.createLinearGradient(72, 52, 124, 104)
  mark.addColorStop(0, '#7fd4ff')
  mark.addColorStop(1, '#2f7de0')
  ctx.fillStyle = mark
  roundRect(ctx, 72, 52, 52, 52, 14)
  ctx.fill()
  ctx.fillStyle = '#06101f'
  ctx.font = `700 30px ${FONT}`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('N', 98, 79)

  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = '#f4f9ff'
  ctx.font = `700 34px ${FONT}`
  ctx.fillText('NeuroKiln', 140, 87)
  ctx.fillStyle = 'rgba(214, 232, 252, 0.72)'
  ctx.font = `400 18px ${FONT}`
  ctx.fillText(label('脑炉 · BCI 算法评测', 'Brain furnace · BCI evaluation'), 140, 112)

  ctx.textAlign = 'right'
  ctx.fillStyle = 'rgba(190, 219, 250, 0.78)'
  ctx.font = `600 20px ${FONT}`
  ctx.fillText(label(`实验 #${detail.id}`, `Experiment #${detail.id}`), WIDTH - 72, 84)

  const name = tr(detail.name) || label('未命名实验', 'Untitled experiment')
  ctx.textAlign = 'left'
  ctx.fillStyle = '#ffffff'
  ctx.font = `700 58px ${FONT}`
  ctx.fillText(clipText(ctx, name, WIDTH - 144), 72, 226)

  const dataset = detail.dataset || {}
  ctx.fillStyle = 'rgba(198, 224, 250, 0.78)'
  ctx.font = `400 22px ${FONT}`
  ctx.fillText(
    `${tr(dataset.name) || dataset.id || '—'} · ${detail.protocol || '—'} · ${folds || '—'} ${label('折', 'folds')}`,
    72,
    262,
  )

  const metrics = [
    ['Balanced Accuracy', formatNumber(best?.balanced_accuracy_mean)],
    ['Cohen κ', formatNumber(best?.cohen_kappa_mean)],
    [label('参评丹方', 'Pipelines'), String(rows.length)],
  ]
  metrics.forEach(([label, value], index) => {
    const x = 72 + index * 248
    ctx.fillStyle = 'rgba(10, 26, 44, 0.58)'
    roundRect(ctx, x, 300, 224, 112, 18)
    ctx.fill()
    ctx.strokeStyle = 'rgba(126, 186, 255, 0.22)'
    ctx.lineWidth = 1
    roundRect(ctx, x, 300, 224, 112, 18)
    ctx.stroke()
    ctx.fillStyle = 'rgba(174, 208, 242, 0.85)'
    ctx.font = `400 17px ${FONT}`
    ctx.fillText(label, x + 20, 330)
    ctx.fillStyle = '#ffffff'
    ctx.font = `700 40px ${FONT}`
    ctx.fillText(value, x + 20, 382)
  })

  if (best) {
    ctx.fillStyle = 'rgba(216, 236, 255, 0.9)'
    ctx.font = `500 20px ${FONT}`
    ctx.fillText(`${label('最佳丹方', 'Best pipeline')} · ${clipText(ctx, tr(best.pipeline_name) || best.pipeline || '—', 640)}`, 72, 462)
  }

  const pills = [
    label('防泄漏拟合', 'Leakage-safe'),
    label('协议统一', 'Unified protocol'),
    label('种子可复现', 'Reproducible seeds'),
  ]
  let pillX = 72
  pills.forEach((label) => {
    ctx.font = `500 17px ${FONT}`
    const width = ctx.measureText(label).width + 32
    ctx.fillStyle = 'rgba(85, 160, 235, 0.18)'
    roundRect(ctx, pillX, 488, width, 36, 18)
    ctx.fill()
    ctx.fillStyle = 'rgba(198, 228, 255, 0.92)'
    ctx.fillText(label, pillX + 16, 512)
    pillX += width + 12
  })

  const url = demoUrl()
  const qrSize = 132
  const qrX = WIDTH - 72 - qrSize
  const qrY = 446 - 12
  const qrData = await QRCode.toDataURL(url, {
    width: 320,
    margin: 1,
    errorCorrectionLevel: 'M',
    color: { dark: '#0a1424', light: '#ffffff' },
  })
  const qrImage = await loadImage(qrData)
  ctx.save()
  roundRect(ctx, qrX, qrY, qrSize, qrSize, 14)
  ctx.clip()
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(qrX, qrY, qrSize, qrSize)
  ctx.drawImage(qrImage, qrX + 6, qrY + 6, qrSize - 12, qrSize - 12)
  ctx.restore()

  ctx.textAlign = 'right'
  ctx.fillStyle = 'rgba(190, 219, 250, 0.8)'
  ctx.font = `500 17px ${FONT}`
  ctx.fillText(label('扫码打开只读演示', 'Scan for the read-only demo'), qrX - 20, qrY + 46)
  ctx.fillStyle = 'rgba(160, 196, 232, 0.62)'
  ctx.font = `400 14px ${FONT}`
  ctx.fillText(clipText(ctx, url, 320), qrX - 20, qrY + 72)

  ctx.textAlign = 'left'
  ctx.fillStyle = 'rgba(160, 196, 232, 0.6)'
  ctx.font = `400 15px ${FONT}`
  ctx.fillText(label('由 NeuroKiln 生成 · 所有统计量仅在训练折拟合', 'Generated by NeuroKiln · all statistics fitted on training folds only'), 72, 590)

  return canvas.toDataURL('image/png')
}

export function downloadShareCard(dataUrl, experimentId) {
  const link = document.createElement('a')
  link.href = dataUrl
  link.download = `neurokiln-share-${experimentId}.png`
  link.click()
}
