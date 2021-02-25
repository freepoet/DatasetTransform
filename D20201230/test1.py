import torch
def nms(bboxes,scores,threshold=0.5):
    x1=bboxes[:,0]
    y1=bboxes[:,1]
    x2=bboxes[:,2]
    y2=bboxes[:,3]

    areas=abs((x2-x1+1)*(y2-y1+1))
    _,order=scores.sort(0,descending=True)
    #order是下标
    keep=[]
    # numel()返回参数个数
    # params = sum(p.numel() for p in list(net.parameters())) / 1e6  # numel()
    # print('#Params: %.1fM' % (params)
    # item() 返回可遍历的(键, 值)元组数组。

    while order.numel()>0:
        if order.numel()==1:
            i=order.item()
            keep.append(i)
            break
        else:
            i=order[0]

            keep.append(i)
        xx1=x1[order[1:]].clamp(min=x1[i])
        yy1=y1[order[1:]].clamp(min=y1[i])
        xx2=x2[order[1:]].clamp(max=x2[i])
        yy2=y2[order[1:]].clamp(max=y2[i])

        inter=(xx2-xx1).clamp(min=0)*(yy2-yy1).clamp(min=0)

        iou=inter/(areas[i]+areas[order[1:]]-inter)

        idx=(iou<=threshold).nonzero().squeeze()
        if idx.numel()==0:
            break
        order=order[idx+1]
    return torch.LongTensor(keep)
bboxes=torch.rand(10,4)*100
print(bboxes)
scores=torch.rand(10,1)
print(scores)
out=nms(bboxes,scores,threshold=0.5)
print(out)