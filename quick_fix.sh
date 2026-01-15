#!/bin/bash
# 快速权限修复

sudo chown -R jovyan:jovyan /home/jovyan/JQ/gad_gspo_B300/models
sudo chmod -R 755 /home/jovyan/JQ/gad_gspo_B300/models
echo "权限修复完成！"